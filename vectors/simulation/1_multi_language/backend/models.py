"""
Database models and data access layer for the e-commerce application.
Provides advanced querying capabilities and business logic.
"""

from sqlalchemy import and_, or_, func, text, case
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy.sql import exists
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from app import User, Product, Category, Order, OrderItem, Review

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

@dataclass
class ProductAnalytics:
    """Product analytics data structure."""
    product_id: int
    product_name: str
    total_sales: float
    units_sold: int
    average_rating: float
    review_count: int
    revenue_rank: int

@dataclass
class UserAnalytics:
    """User analytics data structure."""
    user_id: int
    username: str
    total_orders: int
    total_spent: float
    average_order_value: float
    last_order_date: Optional[datetime]
    customer_lifetime_value: float

class AdvancedUserRepository:
    """Advanced user data access and business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_with_full_profile(self, user_id: int) -> Optional[User]:
        """Get user with all related data loaded."""
        return self.db.query(User).options(
            selectinload(User.orders).selectinload(Order.order_items),
            selectinload(User.reviews).selectinload(Review.product)
        ).filter(User.id == user_id).first()
    
    def get_user_analytics(self, user_id: int) -> Optional[UserAnalytics]:
        """Get comprehensive analytics for a user."""
        result = self.db.query(
            User.id,
            User.username,
            func.count(Order.id).label('total_orders'),
            func.coalesce(func.sum(Order.total_amount), 0).label('total_spent'),
            func.coalesce(func.avg(Order.total_amount), 0).label('average_order_value'),
            func.max(Order.created_at).label('last_order_date')
        ).outerjoin(Order).filter(User.id == user_id).group_by(User.id, User.username).first()
        
        if not result:
            return None
        
        # Calculate customer lifetime value (simplified)
        months_active = 1
        if result.last_order_date:
            first_order = self.db.query(func.min(Order.created_at)).filter(Order.user_id == user_id).scalar()
            if first_order:
                months_active = max(1, (result.last_order_date - first_order).days / 30)
        
        clv = result.total_spent * (months_active / 12) * 1.2  # Simplified CLV calculation
        
        return UserAnalytics(
            user_id=result.id,
            username=result.username,
            total_orders=result.total_orders,
            total_spent=float(result.total_spent),
            average_order_value=float(result.average_order_value),
            last_order_date=result.last_order_date,
            customer_lifetime_value=clv
        )
    
    def get_top_customers(self, limit: int = 10) -> List[UserAnalytics]:
        """Get top customers by total spending."""
        results = self.db.query(
            User.id,
            User.username,
            func.count(Order.id).label('total_orders'),
            func.sum(Order.total_amount).label('total_spent'),
            func.avg(Order.total_amount).label('average_order_value'),
            func.max(Order.created_at).label('last_order_date')
        ).join(Order).group_by(User.id, User.username)\
         .order_by(func.sum(Order.total_amount).desc())\
         .limit(limit).all()
        
        analytics = []
        for result in results:
            months_active = 1
            if result.last_order_date:
                first_order = self.db.query(func.min(Order.created_at)).filter(Order.user_id == result.id).scalar()
                if first_order:
                    months_active = max(1, (result.last_order_date - first_order).days / 30)
            
            clv = result.total_spent * (months_active / 12) * 1.2
            
            analytics.append(UserAnalytics(
                user_id=result.id,
                username=result.username,
                total_orders=result.total_orders,
                total_spent=float(result.total_spent),
                average_order_value=float(result.average_order_value),
                last_order_date=result.last_order_date,
                customer_lifetime_value=clv
            ))
        
        return analytics

class AdvancedProductRepository:
    """Advanced product data access and analytics."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_product_with_reviews(self, product_id: int) -> Optional[Product]:
        """Get product with all reviews and ratings."""
        return self.db.query(Product).options(
            selectinload(Product.reviews).selectinload(Review.user),
            selectinload(Product.category)
        ).filter(Product.id == product_id).first()
    
    def search_products_advanced(
        self, 
        query: str, 
        category_ids: List[int] = None,
        min_price: float = None,
        max_price: float = None,
        min_rating: float = None,
        in_stock_only: bool = False,
        sort_by: str = "relevance"
    ) -> List[Product]:
        """Advanced product search with multiple filters."""
        
        # Base query with search
        base_query = self.db.query(Product).filter(Product.is_active == True)
        
        if query:
            search_filter = or_(
                Product.name.ilike(f"%{query}%"),
                Product.description.ilike(f"%{query}%"),
                Product.sku.ilike(f"%{query}%")
            )
            base_query = base_query.filter(search_filter)
        
        # Category filter
        if category_ids:
            base_query = base_query.filter(Product.category_id.in_(category_ids))
        
        # Price filters
        if min_price is not None:
            base_query = base_query.filter(Product.price >= min_price)
        if max_price is not None:
            base_query = base_query.filter(Product.price <= max_price)
        
        # Stock filter
        if in_stock_only:
            base_query = base_query.filter(Product.stock_quantity > 0)
        
        # Rating filter (requires subquery)
        if min_rating is not None:
            rating_subquery = self.db.query(
                Review.product_id,
                func.avg(Review.rating).label('avg_rating')
            ).group_by(Review.product_id).subquery()
            
            base_query = base_query.join(
                rating_subquery, 
                Product.id == rating_subquery.c.product_id
            ).filter(rating_subquery.c.avg_rating >= min_rating)
        
        # Sorting
        if sort_by == "price_asc":
            base_query = base_query.order_by(Product.price.asc())
        elif sort_by == "price_desc":
            base_query = base_query.order_by(Product.price.desc())
        elif sort_by == "name":
            base_query = base_query.order_by(Product.name.asc())
        elif sort_by == "newest":
            base_query = base_query.order_by(Product.created_at.desc())
        elif sort_by == "rating":
            # Complex sorting by average rating
            rating_subquery = self.db.query(
                Review.product_id,
                func.avg(Review.rating).label('avg_rating')
            ).group_by(Review.product_id).subquery()
            
            base_query = base_query.outerjoin(
                rating_subquery, 
                Product.id == rating_subquery.c.product_id
            ).order_by(
                func.coalesce(rating_subquery.c.avg_rating, 0).desc()
            )
        
        return base_query.all()
    
    def get_product_analytics(self, product_id: int) -> Optional[ProductAnalytics]:
        """Get comprehensive analytics for a product."""
        result = self.db.query(
            Product.id,
            Product.name,
            func.coalesce(func.sum(OrderItem.total_price), 0).label('total_sales'),
            func.coalesce(func.sum(OrderItem.quantity), 0).label('units_sold'),
            func.coalesce(func.avg(Review.rating), 0).label('average_rating'),
            func.count(Review.id).label('review_count')
        ).outerjoin(OrderItem).outerjoin(Review)\
         .filter(Product.id == product_id)\
         .group_by(Product.id, Product.name).first()
        
        if not result:
            return None
        
        # Get revenue rank
        revenue_rank_query = self.db.query(
            func.row_number().over(
                order_by=func.sum(OrderItem.total_price).desc()
            ).label('rank')
        ).select_from(Product)\
         .outerjoin(OrderItem)\
         .filter(Product.id == product_id)\
         .group_by(Product.id)
        
        revenue_rank = revenue_rank_query.scalar() or 0
        
        return ProductAnalytics(
            product_id=result.id,
            product_name=result.name,
            total_sales=float(result.total_sales),
            units_sold=result.units_sold,
            average_rating=float(result.average_rating),
            review_count=result.review_count,
            revenue_rank=revenue_rank
        )
    
    def get_trending_products(self, days: int = 30, limit: int = 10) -> List[ProductAnalytics]:
        """Get trending products based on recent sales."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        results = self.db.query(
            Product.id,
            Product.name,
            func.sum(OrderItem.total_price).label('recent_sales'),
            func.sum(OrderItem.quantity).label('recent_units'),
            func.coalesce(func.avg(Review.rating), 0).label('average_rating'),
            func.count(Review.id).label('review_count')
        ).join(OrderItem).join(Order)\
         .outerjoin(Review, Product.id == Review.product_id)\
         .filter(Order.created_at >= cutoff_date)\
         .group_by(Product.id, Product.name)\
         .order_by(func.sum(OrderItem.total_price).desc())\
         .limit(limit).all()
        
        analytics = []
        for i, result in enumerate(results, 1):
            analytics.append(ProductAnalytics(
                product_id=result.id,
                product_name=result.name,
                total_sales=float(result.recent_sales),
                units_sold=result.recent_units,
                average_rating=float(result.average_rating),
                review_count=result.review_count,
                revenue_rank=i
            ))
        
        return analytics

class AdvancedOrderRepository:
    """Advanced order management and analytics."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_order_with_details(self, order_id: int) -> Optional[Order]:
        """Get order with all related data."""
        return self.db.query(Order).options(
            selectinload(Order.order_items).selectinload(OrderItem.product),
            selectinload(Order.user)
        ).filter(Order.id == order_id).first()
    
    def get_orders_by_status_and_date(
        self, 
        status: OrderStatus = None,
        start_date: datetime = None,
        end_date: datetime = None,
        user_id: int = None
    ) -> List[Order]:
        """Get orders filtered by status and date range."""
        query = self.db.query(Order)
        
        if status:
            query = query.filter(Order.status == status.value)
        
        if start_date:
            query = query.filter(Order.created_at >= start_date)
        
        if end_date:
            query = query.filter(Order.created_at <= end_date)
        
        if user_id:
            query = query.filter(Order.user_id == user_id)
        
        return query.order_by(Order.created_at.desc()).all()
    
    def update_order_status(self, order_id: int, new_status: OrderStatus) -> bool:
        """Update order status with validation."""
        order = self.db.query(Order).filter(Order.id == order_id).first()
        if not order:
            return False
        
        # Status transition validation
        valid_transitions = {
            OrderStatus.PENDING: [OrderStatus.CONFIRMED, OrderStatus.CANCELLED],
            OrderStatus.CONFIRMED: [OrderStatus.PROCESSING, OrderStatus.CANCELLED],
            OrderStatus.PROCESSING: [OrderStatus.SHIPPED, OrderStatus.CANCELLED],
            OrderStatus.SHIPPED: [OrderStatus.DELIVERED],
            OrderStatus.DELIVERED: [OrderStatus.REFUNDED],
            OrderStatus.CANCELLED: [],
            OrderStatus.REFUNDED: []
        }
        
        current_status = OrderStatus(order.status)
        if new_status not in valid_transitions.get(current_status, []):
            logger.warning(f"Invalid status transition from {current_status} to {new_status}")
            return False
        
        order.status = new_status.value
        order.updated_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"Order {order_id} status updated to {new_status.value}")
        return True
    
    def get_sales_analytics(
        self, 
        start_date: datetime = None, 
        end_date: datetime = None
    ) -> Dict[str, Any]:
        """Get comprehensive sales analytics."""
        query = self.db.query(Order)
        
        if start_date:
            query = query.filter(Order.created_at >= start_date)
        if end_date:
            query = query.filter(Order.created_at <= end_date)
        
        # Basic metrics
        total_orders = query.count()
        total_revenue = query.with_entities(func.sum(Order.total_amount)).scalar() or 0
        average_order_value = query.with_entities(func.avg(Order.total_amount)).scalar() or 0
        
        # Status breakdown
        status_breakdown = self.db.query(
            Order.status,
            func.count(Order.id).label('count'),
            func.sum(Order.total_amount).label('revenue')
        ).group_by(Order.status)
        
        if start_date:
            status_breakdown = status_breakdown.filter(Order.created_at >= start_date)
        if end_date:
            status_breakdown = status_breakdown.filter(Order.created_at <= end_date)
        
        status_data = {
            row.status: {"count": row.count, "revenue": float(row.revenue or 0)}
            for row in status_breakdown.all()
        }
        
        # Daily sales trend (last 30 days)
        daily_sales = self.db.query(
            func.date(Order.created_at).label('date'),
            func.count(Order.id).label('orders'),
            func.sum(Order.total_amount).label('revenue')
        ).filter(
            Order.created_at >= datetime.utcnow() - timedelta(days=30)
        ).group_by(func.date(Order.created_at)).all()
        
        return {
            "total_orders": total_orders,
            "total_revenue": float(total_revenue),
            "average_order_value": float(average_order_value),
            "status_breakdown": status_data,
            "daily_sales": [
                {
                    "date": str(row.date),
                    "orders": row.orders,
                    "revenue": float(row.revenue)
                }
                for row in daily_sales
            ]
        }

class RecommendationEngine:
    """Product recommendation engine using collaborative filtering."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_recommendations(self, user_id: int, limit: int = 10) -> List[Product]:
        """Get product recommendations for a user based on purchase history."""
        # Get user's purchase history
        user_products = self.db.query(Product.id).join(OrderItem).join(Order)\
                              .filter(Order.user_id == user_id).distinct().subquery()
        
        # Find similar users (users who bought similar products)
        similar_users = self.db.query(Order.user_id)\
                               .join(OrderItem)\
                               .filter(OrderItem.product_id.in_(user_products))\
                               .filter(Order.user_id != user_id)\
                               .group_by(Order.user_id)\
                               .having(func.count(OrderItem.product_id) >= 2)\
                               .subquery()
        
        # Get products bought by similar users but not by current user
        recommended_products = self.db.query(Product)\
                                     .join(OrderItem)\
                                     .join(Order)\
                                     .filter(Order.user_id.in_(similar_users))\
                                     .filter(~Product.id.in_(user_products))\
                                     .filter(Product.is_active == True)\
                                     .group_by(Product.id)\
                                     .order_by(func.count(OrderItem.id).desc())\
                                     .limit(limit).all()
        
        return recommended_products
    
    def get_product_recommendations(self, product_id: int, limit: int = 5) -> List[Product]:
        """Get products frequently bought together with the given product."""
        # Find orders containing the given product
        orders_with_product = self.db.query(Order.id)\
                                     .join(OrderItem)\
                                     .filter(OrderItem.product_id == product_id)\
                                     .subquery()
        
        # Find other products in those orders
        related_products = self.db.query(Product)\
                                  .join(OrderItem)\
                                  .join(Order)\
                                  .filter(Order.id.in_(orders_with_product))\
                                  .filter(OrderItem.product_id != product_id)\
                                  .filter(Product.is_active == True)\
                                  .group_by(Product.id)\
                                  .order_by(func.count(OrderItem.id).desc())\
                                  .limit(limit).all()
        
        return related_products

# Factory functions for repository creation
def create_user_repository(db: Session) -> AdvancedUserRepository:
    """Create a user repository instance."""
    return AdvancedUserRepository(db)

def create_product_repository(db: Session) -> AdvancedProductRepository:
    """Create a product repository instance."""
    return AdvancedProductRepository(db)

def create_order_repository(db: Session) -> AdvancedOrderRepository:
    """Create an order repository instance."""
    return AdvancedOrderRepository(db)

def create_recommendation_engine(db: Session) -> RecommendationEngine:
    """Create a recommendation engine instance."""
    return RecommendationEngine(db)