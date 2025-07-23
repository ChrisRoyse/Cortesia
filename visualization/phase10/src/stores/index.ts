import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import visualizationSlice from './visualizationSlice';
import systemSlice from './systemSlice';
import componentsSlice from './componentsSlice';
import performanceSlice from './performanceSlice';

export const store = configureStore({
  reducer: {
    visualization: visualizationSlice,
    system: systemSlice,
    components: componentsSlice,
    performance: performanceSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
        ignoredPaths: ['register'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

export * from './visualizationSlice';
export * from './systemSlice';
export * from './componentsSlice';
export * from './performanceSlice';