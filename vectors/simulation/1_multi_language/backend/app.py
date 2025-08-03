"""
FastAPI Backend Application for E-commerce Platform
Provides RESTful APIs for user management, product catalog, and order processing.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import jwt
import os
from contextlib import asynccontextmanager
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost:5432/ecommerce"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class User(Base):
    """User model with authentication and profile information."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    orders = relationship("Order", back_populates="user")
    reviews = relationship("Review", back_populates="user")

class Product(Base):
    """Product model for e-commerce catalog."""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(String(2000))
    price = Column(Float, nullable=False)
    stock_quantity = Column(Integer, default=0)
    category_id = Column(Integer, ForeignKey("categories.id"))
    sku = Column(String(100), unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    category = relationship("Category", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")
    reviews = relationship("Review", back_populates="product")

class Category(Base):
    """Product category model."""
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(String(500))
    parent_id = Column(Integer, ForeignKey("categories.id"))
    is_active = Column(Boolean, default=True)
    
    # Relationships
    products = relationship("Product", back_populates="category")
    children = relationship("Category")

class Order(Base):
    """Order model for purchase tracking."""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    total_amount = Column(Float, nullable=False)
    status = Column(String(50), default="pending")  # pending, confirmed, shipped, delivered, cancelled
    shipping_address = Column(String(1000))
    billing_address = Column(String(1000))
    payment_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    order_items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    """Individual items within an order."""
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="order_items")
    product = relationship("Product", back_populates="order_items")

class Review(Base):
    """Product review model."""
    __tablename__ = "reviews"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    comment = Column(String(2000))
    is_verified_purchase = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="reviews")
    product = relationship("Product", back_populates="reviews")

# Pydantic models for API
class UserBase(BaseModel):
    username: str
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    stock_quantity: int = 0
    category_id: Optional[int] = None
    sku: Optional[str] = None

class ProductCreate(ProductBase):
    pass

class ProductResponse(ProductBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class OrderCreate(BaseModel):
    items: List[Dict[str, Any]]  # [{"product_id": 1, "quantity": 2}, ...]
    shipping_address: str
    billing_address: Optional[str] = None
    payment_method: str

class OrderResponse(BaseModel):
    id: int
    user_id: int
    total_amount: float
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# FastAPI app initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up e-commerce backend...")
    Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    logger.info("Shutting down e-commerce backend...")

app = FastAPI(
    title="E-commerce API",
    description="A comprehensive e-commerce backend API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """Get current authenticated user."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Utility functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    # Create new user
    hashed_password = hash_password(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"New user registered: {user.username}")
    return db_user

@app.post("/auth/login")
async def login_user(username: str, password: str, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is disabled")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in: {username}")
    return {"access_token": access_token, "token_type": "bearer"}

# User management endpoints
@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return current_user

@app.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    users = db.query(User).offset(skip).limit(limit).all()
    return users

# Product management endpoints
@app.get("/products", response_model=List[ProductResponse])
async def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """List products with optional filtering."""
    query = db.query(Product).filter(Product.is_active == True)
    
    if category_id:
        query = query.filter(Product.category_id == category_id)
    
    if search:
        query = query.filter(
            (Product.name.ilike(f"%{search}%")) |
            (Product.description.ilike(f"%{search}%"))
        )
    
    products = query.offset(skip).limit(limit).all()
    return products

@app.get("/products/{product_id}", response_model=ProductResponse)
async def get_product(product_id: int, db: Session = Depends(get_db)):
    """Get a specific product by ID."""
    product = db.query(Product).filter(Product.id == product_id, Product.is_active == True).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@app.post("/products", response_model=ProductResponse)
async def create_product(
    product: ProductCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new product (admin only)."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    db_product = Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    
    logger.info(f"New product created: {product.name}")
    return db_product

# Order management endpoints
@app.get("/orders", response_model=List[OrderResponse])
async def list_user_orders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List current user's orders."""
    orders = db.query(Order).filter(Order.user_id == current_user.id).all()
    return orders

@app.post("/orders", response_model=OrderResponse)
async def create_order(
    order: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new order."""
    # Calculate total amount and validate items
    total_amount = 0.0
    order_items = []
    
    for item in order.items:
        product = db.query(Product).filter(Product.id == item["product_id"]).first()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {item['product_id']} not found")
        
        if product.stock_quantity < item["quantity"]:
            raise HTTPException(status_code=400, detail=f"Insufficient stock for product {product.name}")
        
        item_total = product.price * item["quantity"]
        total_amount += item_total
        
        order_items.append(OrderItem(
            product_id=product.id,
            quantity=item["quantity"],
            unit_price=product.price,
            total_price=item_total
        ))
    
    # Create order
    db_order = Order(
        user_id=current_user.id,
        total_amount=total_amount,
        shipping_address=order.shipping_address,
        billing_address=order.billing_address or order.shipping_address,
        payment_method=order.payment_method,
        order_items=order_items
    )
    
    db.add(db_order)
    db.commit()
    db.refresh(db_order)
    
    logger.info(f"New order created: {db_order.id} for user {current_user.username}")
    return db_order

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)