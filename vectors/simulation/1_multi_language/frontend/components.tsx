/**
 * React TypeScript Components for E-commerce Application
 * Modern React components with TypeScript, hooks, and advanced patterns
 */

import React, { useState, useEffect, useContext, useReducer, useCallback, useMemo, useRef } from 'react';
import { createContext, ReactNode, FC, ComponentType } from 'react';

// Type definitions
interface User {
  id: number;
  username: string;
  email: string;
  firstName?: string;
  lastName?: string;
  isActive: boolean;
  createdAt: Date;
}

interface Product {
  id: number;
  name: string;
  description?: string;
  price: number;
  stockQuantity: number;
  categoryId?: number;
  sku?: string;
  isActive: boolean;
  createdAt: Date;
  imageUrl?: string;
  category?: Category;
  reviews?: Review[];
}

interface Category {
  id: number;
  name: string;
  description?: string;
  parentId?: number;
  isActive: boolean;
  children?: Category[];
}

interface CartItem {
  product: Product;
  quantity: number;
  totalPrice: number;
}

interface Order {
  id: number;
  userId: number;
  totalAmount: number;
  status: OrderStatus;
  shippingAddress: string;
  billingAddress?: string;
  paymentMethod: string;
  createdAt: Date;
  updatedAt: Date;
  items: OrderItem[];
}

interface OrderItem {
  id: number;
  orderId: number;
  productId: number;
  quantity: number;
  unitPrice: number;
  totalPrice: number;
  product?: Product;
}

interface Review {
  id: number;
  userId: number;
  productId: number;
  rating: number;
  comment?: string;
  isVerifiedPurchase: boolean;
  createdAt: Date;
  user?: User;
}

enum OrderStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  PROCESSING = 'processing',
  SHIPPED = 'shipped',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
  REFUNDED = 'refunded'
}

// Context definitions
interface AppContextType {
  user: User | null;
  cart: CartItem[];
  loading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  addToCart: (product: Product, quantity: number) => void;
  removeFromCart: (productId: number) => void;
  updateCartQuantity: (productId: number, quantity: number) => void;
  clearCart: () => void;
  setError: (error: string | null) => void;
  setLoading: (loading: boolean) => void;
}

interface ProductContextType {
  products: Product[];
  categories: Category[];
  currentProduct: Product | null;
  searchResults: Product[];
  loading: boolean;
  searchProducts: (query: string, filters?: ProductFilters) => Promise<void>;
  getProduct: (id: number) => Promise<Product | null>;
  getFeaturedProducts: () => Promise<Product[]>;
  getProductsByCategory: (categoryId: number) => Promise<Product[]>;
}

interface ProductFilters {
  categoryIds?: number[];
  minPrice?: number;
  maxPrice?: number;
  minRating?: number;
  inStockOnly?: boolean;
  sortBy?: 'name' | 'price_asc' | 'price_desc' | 'newest' | 'rating';
}

// State management with useReducer
interface AppState {
  user: User | null;
  cart: CartItem[];
  loading: boolean;
  error: string | null;
}

type AppAction =
  | { type: 'SET_USER'; payload: User | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'ADD_TO_CART'; payload: { product: Product; quantity: number } }
  | { type: 'REMOVE_FROM_CART'; payload: number }
  | { type: 'UPDATE_CART_QUANTITY'; payload: { productId: number; quantity: number } }
  | { type: 'CLEAR_CART' };

const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_USER':
      return { ...state, user: action.payload };
    
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    case 'ADD_TO_CART': {
      const { product, quantity } = action.payload;
      const existingItem = state.cart.find(item => item.product.id === product.id);
      
      if (existingItem) {
        return {
          ...state,
          cart: state.cart.map(item =>
            item.product.id === product.id
              ? {
                  ...item,
                  quantity: item.quantity + quantity,
                  totalPrice: (item.quantity + quantity) * product.price
                }
              : item
          )
        };
      }
      
      return {
        ...state,
        cart: [
          ...state.cart,
          {
            product,
            quantity,
            totalPrice: quantity * product.price
          }
        ]
      };
    }
    
    case 'REMOVE_FROM_CART':
      return {
        ...state,
        cart: state.cart.filter(item => item.product.id !== action.payload)
      };
    
    case 'UPDATE_CART_QUANTITY': {
      const { productId, quantity } = action.payload;
      
      if (quantity <= 0) {
        return {
          ...state,
          cart: state.cart.filter(item => item.product.id !== productId)
        };
      }
      
      return {
        ...state,
        cart: state.cart.map(item =>
          item.product.id === productId
            ? {
                ...item,
                quantity,
                totalPrice: quantity * item.product.price
              }
            : item
        )
      };
    }
    
    case 'CLEAR_CART':
      return { ...state, cart: [] };
    
    default:
      return state;
  }
};

// Contexts
const AppContext = createContext<AppContextType | undefined>(undefined);
const ProductContext = createContext<ProductContextType | undefined>(undefined);

// Custom hooks
const useAppContext = (): AppContextType => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

const useProductContext = (): ProductContextType => {
  const context = useContext(ProductContext);
  if (!context) {
    throw new Error('useProductContext must be used within a ProductProvider');
  }
  return context;
};

// HOC for authentication
const withAuth = <P extends object>(Component: ComponentType<P>) => {
  return (props: P) => {
    const { user } = useAppContext();
    
    if (!user) {
      return (
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>Please log in to access this page.</p>
        </div>
      );
    }
    
    return <Component {...props} />;
  };
};

// Custom hooks for data fetching
const useProducts = (filters?: ProductFilters) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fetchProducts = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Simulate API call
      const response = await fetch('/api/products?' + new URLSearchParams({
        ...(filters?.categoryIds && { categoryIds: filters.categoryIds.join(',') }),
        ...(filters?.minPrice && { minPrice: filters.minPrice.toString() }),
        ...(filters?.maxPrice && { maxPrice: filters.maxPrice.toString() }),
        ...(filters?.sortBy && { sortBy: filters.sortBy })
      }));
      
      if (!response.ok) {
        throw new Error('Failed to fetch products');
      }
      
      const data = await response.json();
      setProducts(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [filters]);
  
  useEffect(() => {
    fetchProducts();
  }, [fetchProducts]);
  
  return { products, loading, error, refetch: fetchProducts };
};

// Provider Components
const AppProvider: FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, {
    user: null,
    cart: [],
    loading: false,
    error: null
  });
  
  const login = useCallback(async (username: string, password: string) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      
      if (!response.ok) {
        throw new Error('Invalid credentials');
      }
      
      const { access_token, user } = await response.json();
      localStorage.setItem('authToken', access_token);
      dispatch({ type: 'SET_USER', payload: user });
    } catch (error) {
      dispatch({ 
        type: 'SET_ERROR', 
        payload: error instanceof Error ? error.message : 'Login failed' 
      });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);
  
  const logout = useCallback(() => {
    localStorage.removeItem('authToken');
    dispatch({ type: 'SET_USER', payload: null });
    dispatch({ type: 'CLEAR_CART' });
  }, []);
  
  const addToCart = useCallback((product: Product, quantity: number) => {
    dispatch({ type: 'ADD_TO_CART', payload: { product, quantity } });
  }, []);
  
  const removeFromCart = useCallback((productId: number) => {
    dispatch({ type: 'REMOVE_FROM_CART', payload: productId });
  }, []);
  
  const updateCartQuantity = useCallback((productId: number, quantity: number) => {
    dispatch({ type: 'UPDATE_CART_QUANTITY', payload: { productId, quantity } });
  }, []);
  
  const clearCart = useCallback(() => {
    dispatch({ type: 'CLEAR_CART' });
  }, []);
  
  const setError = useCallback((error: string | null) => {
    dispatch({ type: 'SET_ERROR', payload: error });
  }, []);
  
  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  }, []);
  
  const contextValue = useMemo(() => ({
    user: state.user,
    cart: state.cart,
    loading: state.loading,
    error: state.error,
    login,
    logout,
    addToCart,
    removeFromCart,
    updateCartQuantity,
    clearCart,
    setError,
    setLoading
  }), [state, login, logout, addToCart, removeFromCart, updateCartQuantity, clearCart, setError, setLoading]);
  
  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
};

// Component definitions
const Header: FC = () => {
  const { user, cart, logout } = useAppContext();
  const cartItemCount = cart.reduce((total, item) => total + item.quantity, 0);
  const cartTotal = cart.reduce((total, item) => total + item.totalPrice, 0);
  
  return (
    <header className="app-header">
      <div className="header-content">
        <h1 className="app-logo">E-Commerce Store</h1>
        
        <nav className="main-nav">
          <a href="/">Home</a>
          <a href="/products">Products</a>
          {user && <a href="/orders">Orders</a>}
        </nav>
        
        <div className="header-actions">
          <div className="cart-info">
            <button className="cart-button">
              🛒 Cart ({cartItemCount}) - ${cartTotal.toFixed(2)}
            </button>
          </div>
          
          {user ? (
            <div className="user-menu">
              <span>Hello, {user.firstName || user.username}</span>
              <button onClick={logout}>Logout</button>
            </div>
          ) : (
            <div className="auth-buttons">
              <a href="/login">Login</a>
              <a href="/register">Register</a>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

interface ProductCardProps {
  product: Product;
  onAddToCart?: (product: Product, quantity: number) => void;
  showFullDescription?: boolean;
}

const ProductCard: FC<ProductCardProps> = ({ 
  product, 
  onAddToCart, 
  showFullDescription = false 
}) => {
  const { addToCart } = useAppContext();
  const [quantity, setQuantity] = useState(1);
  const [imageError, setImageError] = useState(false);
  
  const handleAddToCart = useCallback(() => {
    const handler = onAddToCart || addToCart;
    handler(product, quantity);
  }, [onAddToCart, addToCart, product, quantity]);
  
  const averageRating = useMemo(() => {
    if (!product.reviews || product.reviews.length === 0) return 0;
    const sum = product.reviews.reduce((acc, review) => acc + review.rating, 0);
    return sum / product.reviews.length;
  }, [product.reviews]);
  
  return (
    <div className="product-card">
      <div className="product-image">
        {product.imageUrl && !imageError ? (
          <img 
            src={product.imageUrl} 
            alt={product.name}
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="placeholder-image">No Image</div>
        )}
      </div>
      
      <div className="product-info">
        <h3 className="product-name">{product.name}</h3>
        
        {product.description && (
          <p className="product-description">
            {showFullDescription 
              ? product.description 
              : product.description.substring(0, 100) + '...'
            }
          </p>
        )}
        
        <div className="product-meta">
          <span className="product-price">${product.price.toFixed(2)}</span>
          
          {product.reviews && product.reviews.length > 0 && (
            <div className="product-rating">
              <span className="rating-stars">
                {'★'.repeat(Math.round(averageRating))}
                {'☆'.repeat(5 - Math.round(averageRating))}
              </span>
              <span className="rating-text">
                ({averageRating.toFixed(1)}) {product.reviews.length} reviews
              </span>
            </div>
          )}
          
          <div className="stock-info">
            {product.stockQuantity > 0 ? (
              <span className="in-stock">In Stock ({product.stockQuantity})</span>
            ) : (
              <span className="out-of-stock">Out of Stock</span>
            )}
          </div>
        </div>
        
        <div className="product-actions">
          <div className="quantity-selector">
            <label htmlFor={`quantity-${product.id}`}>Qty:</label>
            <select 
              id={`quantity-${product.id}`}
              value={quantity} 
              onChange={(e) => setQuantity(parseInt(e.target.value))}
              disabled={product.stockQuantity === 0}
            >
              {Array.from({ length: Math.min(10, product.stockQuantity) }, (_, i) => (
                <option key={i + 1} value={i + 1}>{i + 1}</option>
              ))}
            </select>
          </div>
          
          <button 
            className="add-to-cart-btn"
            onClick={handleAddToCart}
            disabled={product.stockQuantity === 0}
          >
            {product.stockQuantity === 0 ? 'Out of Stock' : 'Add to Cart'}
          </button>
        </div>
      </div>
    </div>
  );
};

interface CartItemProps {
  item: CartItem;
  onUpdateQuantity: (productId: number, quantity: number) => void;
  onRemove: (productId: number) => void;
}

const CartItemComponent: FC<CartItemProps> = ({ item, onUpdateQuantity, onRemove }) => {
  const handleQuantityChange = useCallback((newQuantity: number) => {
    onUpdateQuantity(item.product.id, newQuantity);
  }, [item.product.id, onUpdateQuantity]);
  
  const handleRemove = useCallback(() => {
    onRemove(item.product.id);
  }, [item.product.id, onRemove]);
  
  return (
    <div className="cart-item">
      <div className="item-info">
        <h4>{item.product.name}</h4>
        <p className="item-price">${item.product.price.toFixed(2)} each</p>
      </div>
      
      <div className="quantity-controls">
        <button 
          onClick={() => handleQuantityChange(item.quantity - 1)}
          disabled={item.quantity <= 1}
        >
          -
        </button>
        <span className="quantity">{item.quantity}</span>
        <button 
          onClick={() => handleQuantityChange(item.quantity + 1)}
          disabled={item.quantity >= item.product.stockQuantity}
        >
          +
        </button>
      </div>
      
      <div className="item-total">
        ${item.totalPrice.toFixed(2)}
      </div>
      
      <button className="remove-btn" onClick={handleRemove}>
        Remove
      </button>
    </div>
  );
};

const ShoppingCartComponent: FC = () => {
  const { cart, updateCartQuantity, removeFromCart, clearCart } = useAppContext();
  
  const cartTotal = useMemo(() => 
    cart.reduce((total, item) => total + item.totalPrice, 0), 
    [cart]
  );
  
  if (cart.length === 0) {
    return (
      <div className="empty-cart">
        <h2>Your cart is empty</h2>
        <p>Add some products to get started!</p>
        <a href="/products" className="continue-shopping-btn">
          Continue Shopping
        </a>
      </div>
    );
  }
  
  return (
    <div className="shopping-cart">
      <h2>Shopping Cart</h2>
      
      <div className="cart-items">
        {cart.map(item => (
          <CartItemComponent
            key={item.product.id}
            item={item}
            onUpdateQuantity={updateCartQuantity}
            onRemove={removeFromCart}
          />
        ))}
      </div>
      
      <div className="cart-summary">
        <div className="cart-total">
          <strong>Total: ${cartTotal.toFixed(2)}</strong>
        </div>
        
        <div className="cart-actions">
          <button onClick={clearCart} className="clear-cart-btn">
            Clear Cart
          </button>
          <button className="checkout-btn">
            Proceed to Checkout
          </button>
        </div>
      </div>
    </div>
  );
};

interface ProductFiltersProps {
  filters: ProductFilters;
  onFiltersChange: (filters: ProductFilters) => void;
  categories: Category[];
}

const ProductFilters: FC<ProductFiltersProps> = ({ 
  filters, 
  onFiltersChange, 
  categories 
}) => {
  const handleFilterChange = useCallback((key: keyof ProductFilters, value: any) => {
    onFiltersChange({ ...filters, [key]: value });
  }, [filters, onFiltersChange]);
  
  return (
    <div className="product-filters">
      <h3>Filters</h3>
      
      <div className="filter-group">
        <label>Category:</label>
        <select 
          multiple
          value={filters.categoryIds || []}
          onChange={(e) => {
            const values = Array.from(e.target.selectedOptions, option => parseInt(option.value));
            handleFilterChange('categoryIds', values);
          }}
        >
          {categories.map(category => (
            <option key={category.id} value={category.id}>
              {category.name}
            </option>
          ))}
        </select>
      </div>
      
      <div className="filter-group">
        <label>Price Range:</label>
        <input 
          type="number" 
          placeholder="Min Price"
          value={filters.minPrice || ''}
          onChange={(e) => handleFilterChange('minPrice', parseFloat(e.target.value) || undefined)}
        />
        <input 
          type="number" 
          placeholder="Max Price"
          value={filters.maxPrice || ''}
          onChange={(e) => handleFilterChange('maxPrice', parseFloat(e.target.value) || undefined)}
        />
      </div>
      
      <div className="filter-group">
        <label>
          <input 
            type="checkbox"
            checked={filters.inStockOnly || false}
            onChange={(e) => handleFilterChange('inStockOnly', e.target.checked)}
          />
          In Stock Only
        </label>
      </div>
      
      <div className="filter-group">
        <label>Sort By:</label>
        <select 
          value={filters.sortBy || 'name'}
          onChange={(e) => handleFilterChange('sortBy', e.target.value as ProductFilters['sortBy'])}
        >
          <option value="name">Name</option>
          <option value="price_asc">Price: Low to High</option>
          <option value="price_desc">Price: High to Low</option>
          <option value="newest">Newest First</option>
          <option value="rating">Highest Rated</option>
        </select>
      </div>
    </div>
  );
};

const ProductList: FC = () => {
  const [filters, setFilters] = useState<ProductFilters>({});
  const [searchQuery, setSearchQuery] = useState('');
  const { products, loading, error } = useProducts(filters);
  const searchTimeoutRef = useRef<NodeJS.Timeout>();
  
  const filteredProducts = useMemo(() => {
    if (!searchQuery) return products;
    
    return products.filter(product =>
      product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (product.description && product.description.toLowerCase().includes(searchQuery.toLowerCase()))
    );
  }, [products, searchQuery]);
  
  const handleSearchChange = useCallback((query: string) => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    
    searchTimeoutRef.current = setTimeout(() => {
      setSearchQuery(query);
    }, 300); // Debounce search
  }, []);
  
  useEffect(() => {
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, []);
  
  if (loading) {
    return <div className="loading">Loading products...</div>;
  }
  
  if (error) {
    return <div className="error">Error: {error}</div>;
  }
  
  return (
    <div className="product-list-container">
      <div className="search-bar">
        <input 
          type="text"
          placeholder="Search products..."
          onChange={(e) => handleSearchChange(e.target.value)}
        />
      </div>
      
      <div className="products-content">
        <ProductFilters 
          filters={filters}
          onFiltersChange={setFilters}
          categories={[]} // Would be passed from context
        />
        
        <div className="products-grid">
          {filteredProducts.map(product => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </div>
    </div>
  );
};

// Protected component example
const OrderHistory = withAuth(() => {
  const { user } = useAppContext();
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    const fetchOrders = async () => {
      try {
        const response = await fetch('/api/orders', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('authToken')}`
          }
        });
        
        if (response.ok) {
          const orderData = await response.json();
          setOrders(orderData);
        }
      } catch (error) {
        console.error('Failed to fetch orders:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchOrders();
  }, []);
  
  if (loading) return <div>Loading orders...</div>;
  
  return (
    <div className="order-history">
      <h2>Order History</h2>
      {orders.length === 0 ? (
        <p>No orders found.</p>
      ) : (
        <div className="orders-list">
          {orders.map(order => (
            <div key={order.id} className="order-card">
              <h3>Order #{order.id}</h3>
              <p>Status: <span className={`status-${order.status}`}>{order.status}</span></p>
              <p>Total: ${order.totalAmount.toFixed(2)}</p>
              <p>Date: {new Date(order.createdAt).toLocaleDateString()}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

// Main App component
const App: FC = () => {
  return (
    <AppProvider>
      <div className="app">
        <Header />
        <main className="main-content">
          {/* Router would go here in a real app */}
          <ProductList />
        </main>
      </div>
    </AppProvider>
  );
};

export default App;
export { 
  AppProvider, 
  Header, 
  ProductCard, 
  ShoppingCartComponent, 
  ProductList, 
  OrderHistory,
  useAppContext,
  useProductContext,
  withAuth
};