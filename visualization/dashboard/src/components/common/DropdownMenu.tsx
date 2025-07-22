import React from 'react';

interface MenuItem {
  id: string;
  label: string;
  icon?: React.ComponentType<any>;
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}

interface Divider {
  type: 'divider';
}

type MenuItemType = MenuItem | Divider;

interface DropdownMenuProps {
  items: MenuItemType[];
  onClose: () => void;
  className?: string;
}

export const DropdownMenu: React.FC<DropdownMenuProps> = ({
  items,
  onClose,
  className = "",
}) => {
  const handleItemClick = (item: MenuItem) => {
    if (!item.disabled) {
      item.onClick();
      onClose();
    }
  };

  return (
    <div className={`bg-white rounded-lg shadow-lg border border-gray-200 py-2 min-w-48 ${className}`}>
      {items.map((item, index) => {
        if ('type' in item && item.type === 'divider') {
          return <div key={`divider-${index}`} className="border-t border-gray-200 my-1" />;
        }

        const menuItem = item as MenuItem;
        
        return (
          <button
            key={menuItem.id}
            onClick={() => handleItemClick(menuItem)}
            disabled={menuItem.disabled}
            className={`
              w-full flex items-center px-4 py-2 text-sm text-left
              ${menuItem.disabled 
                ? 'text-gray-400 cursor-not-allowed' 
                : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
              }
              ${menuItem.className || ''}
            `}
          >
            {menuItem.icon && (
              <menuItem.icon className="w-4 h-4 mr-3 flex-shrink-0" />
            )}
            <span>{menuItem.label}</span>
          </button>
        );
      })}
    </div>
  );
};

export default DropdownMenu;