# LLMKG Dashboard Routing & Navigation System

This comprehensive routing and navigation system provides an intuitive, brain-inspired interface for accessing all LLMKG visualization features.

## 🏗️ Architecture Overview

The routing system is built with the following key components:

### Core Routing Components
- **AppRouter.tsx** - Main application router with lazy loading and route guards
- **DashboardLayout.tsx** - Main layout wrapper with sidebar and header
- **Sidebar.tsx** - Collapsible sidebar navigation with real-time status indicators
- **Header.tsx** - Top navigation with search, notifications, and user menu
- **Breadcrumb.tsx** - Hierarchical breadcrumb navigation

### Page Components
- **DashboardPage.tsx** - Main dashboard overview with system metrics
- **ToolsPage.tsx** - MCP tools catalog and management interface
- **ArchitecturePage.tsx** - System architecture and component health monitoring

## 🧠 Brain-Inspired Organization

The navigation is organized around brain-inspired cognitive patterns:

### Primary Categories
1. **Dashboard** - System overview and key metrics
2. **Cognitive Patterns** - Pattern recognition, inhibitory mechanisms, attention systems
3. **Neural Activity** - Activity heatmaps, connectivity maps, spike analysis
4. **Knowledge Graph** - 3D semantic relationship visualization
5. **Memory Systems** - Performance metrics, consolidation monitoring, usage analytics
6. **MCP Tools** - Model Context Protocol tool catalog and testing
7. **System Architecture** - Component health, dependencies, system monitoring

## 🚀 Key Features

### Routing Features
- **Lazy Loading** - Code splitting for optimal performance with React.lazy()
- **Nested Routes** - Hierarchical navigation for complex cognitive features
- **Route Guards** - Authentication and permission-based access control
- **Deep Linking** - Shareable URLs for specific dashboard states
- **Browser Integration** - Full back/forward button support

### Navigation UX
- **Responsive Design** - Mobile-first design with collapsible sidebar
- **Real-time Indicators** - Live status badges showing system health
- **Search Integration** - Quick access to features and tools
- **Breadcrumbs** - Clear navigation hierarchy
- **Favorites** - Bookmark frequently used views

### LLMKG-Specific Features
- **Component Status** - Real-time health indicators for each system component
- **MCP Integration** - Direct access to tool catalog with execution status
- **Performance Awareness** - Route-specific performance metrics
- **Federation Support** - Multi-instance navigation capabilities

## 📁 File Structure

```
src/
├── routing/
│   └── AppRouter.tsx                 # Main router with lazy loading
├── components/
│   ├── Layout/
│   │   └── DashboardLayout.tsx       # Main layout component
│   ├── Navigation/
│   │   ├── Sidebar.tsx               # Collapsible sidebar navigation
│   │   ├── Header.tsx                # Top navigation header
│   │   └── Breadcrumb.tsx            # Breadcrumb navigation
│   └── common/                       # Shared UI components
├── pages/
│   ├── Dashboard/
│   │   └── DashboardPage.tsx         # Main overview page
│   ├── Tools/
│   │   └── ToolsPage.tsx             # MCP tools interface
│   └── Architecture/
│       └── ArchitecturePage.tsx      # System architecture view
└── hooks/
    ├── useBreakpoint.ts              # Responsive breakpoint hook
    └── useOnClickOutside.ts          # Click outside handler
```

## 🛠️ Usage Examples

### Adding a New Route

```tsx
// In AppRouter.tsx, add a new route
<Route path="/new-feature" element={
  <RouteGuard requiredPermissions={['feature_access']}>
    <Suspense fallback={<PageLoader />}>
      <NewFeaturePage />
    </Suspense>
  </RouteGuard>
} />
```

### Adding Navigation Items

```tsx
// In Sidebar.tsx, add to navigationItems array
{
  id: 'new-feature',
  title: 'New Feature',
  path: '/new-feature',
  icon: NewFeatureIcon,
  description: 'Description of the new feature',
}
```

### Using Route Configuration

```tsx
import { routeConfig, buildDeepLink } from '../routing/AppRouter';

// Navigate programmatically
const cognitiveUrl = routeConfig.cognitive.patterns;

// Build deep links with parameters
const deepLink = buildDeepLink('/cognitive/patterns', { 
  view: 'heatmap', 
  timeRange: '24h' 
});
```

### Custom Breadcrumbs

```tsx
import { EnhancedBreadcrumb } from '../components/Navigation/Breadcrumb';

// Custom breadcrumb items
const customItems = [
  { label: 'Dashboard', path: '/', icon: HomeIcon },
  { label: 'Custom Section', path: '/custom', isActive: true }
];

<EnhancedBreadcrumb 
  customItems={customItems}
  showIcons={true}
  showHome={false}
/>
```

## 🎯 Navigation Patterns

### Cognitive Hierarchy
The navigation follows a hierarchical structure matching cognitive processes:

1. **Overview Level** - High-level system status and metrics
2. **Category Level** - Grouped by brain-inspired functions (cognitive, neural, memory)
3. **Feature Level** - Specific tools and visualizations within each category
4. **Detail Level** - Deep-dive analysis and configuration options

### Status Integration
Each navigation item shows real-time status:

```tsx
// Status types
type ComponentStatus = 'active' | 'warning' | 'error' | 'disabled';

// Status indicators in navigation
<StatusIndicator 
  status={componentStatus} 
  showLabel={true}
  className="ml-2"
/>
```

### Search and Filtering
Global search functionality with context-aware results:

```tsx
// Search integration
const handleSearch = (query: string) => {
  // Search across:
  // - Page titles and descriptions
  // - Tool names and functions
  // - Component names and status
  // - Documentation content
};
```

## 🔒 Security & Permissions

### Route Guards
Implement authentication and authorization:

```tsx
// Route protection
<RouteGuard 
  requiresAuth={true}
  requiredPermissions={['admin', 'system_view']}
>
  <ArchitecturePage />
</RouteGuard>
```

### Permission Levels
- **Public** - Dashboard overview, basic metrics
- **User** - Standard cognitive and neural visualization access
- **Admin** - System architecture, tool management, configuration
- **Developer** - Full MCP tool access, debugging features

## 📱 Responsive Design

The navigation adapts to different screen sizes:

### Desktop (≥1024px)
- Full sidebar always visible
- All navigation features available
- Multi-column layouts

### Tablet (768px-1023px)
- Collapsible sidebar
- Touch-friendly navigation
- Optimized layouts

### Mobile (<768px)
- Overlay sidebar
- Simplified navigation
- Mobile-optimized interactions

## 🚦 Performance Considerations

### Code Splitting
All pages are lazy-loaded to optimize initial bundle size:

```tsx
const CognitivePage = lazy(() => import('../pages/Cognitive/CognitivePage'));
```

### Route Preloading
Critical routes can be preloaded:

```tsx
// Preload important routes
import('../pages/Dashboard/DashboardPage');
```

### Bundle Analysis
- Main bundle: Core routing and layout components
- Page bundles: Individual page components and dependencies
- Shared bundles: Common utilities and hooks

## 🔄 Real-time Updates

### Status Monitoring
Navigation components receive real-time updates:

```tsx
const { systemStatus, componentStatuses } = useRealTimeStatus();

// Update navigation indicators
const getItemStatus = (itemId: string) => {
  return componentStatuses[itemId]?.status || 'active';
};
```

### Notification Integration
Real-time notifications appear in the header:

```tsx
const { notifications } = useRealTimeStatus();
const unreadCount = notifications.filter(n => !n.read).length;
```

## 🎨 Theming & Customization

### Brand Colors
- Primary: Blue gradient (brain-tech aesthetic)
- Secondary: Purple (neural networks)
- Success: Green (system health)
- Warning: Yellow (attention required)
- Error: Red (critical issues)

### Icon System
Consistent iconography using Heroicons:
- Cognitive: CPU/Circuit patterns
- Neural: Network/Connection patterns
- Memory: Database/Storage patterns
- Tools: Utility/Wrench patterns

## 🧪 Testing Strategy

### Route Testing
```tsx
// Test route navigation
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

test('navigates to cognitive patterns page', () => {
  render(
    <MemoryRouter initialEntries={['/cognitive/patterns']}>
      <AppRouter />
    </MemoryRouter>
  );
  // Assert page content
});
```

### Navigation Component Testing
- Sidebar collapse/expand functionality
- Breadcrumb navigation accuracy  
- Search functionality
- Real-time status updates

## 🔮 Future Enhancements

### Planned Features
1. **AI-Powered Navigation** - Smart recommendations based on usage patterns
2. **Workflow Presets** - Saved navigation states for common tasks
3. **Multi-Instance Management** - Federation across multiple LLMKG instances
4. **Advanced Search** - Semantic search across all system content
5. **Customizable Layouts** - User-configurable dashboard arrangements

### Integration Points
- **Voice Navigation** - Voice commands for hands-free operation
- **Keyboard Shortcuts** - Power user navigation shortcuts
- **External APIs** - Integration with external monitoring systems
- **Mobile App** - Native mobile navigation experience

This routing and navigation system provides a solid foundation for the LLMKG dashboard while maintaining the flexibility to grow with the system's evolving needs.