import { configureStore } from '@reduxjs/toolkit';
import toolsReducer from '../features/tools/stores/toolsSlice';

export const store = configureStore({
  reducer: {
    tools: toolsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['tools/addTool', 'tools/updateTool'],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['payload.createdAt', 'payload.updatedAt'],
        // Ignore these paths in the state
        ignoredPaths: ['tools.tools'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;