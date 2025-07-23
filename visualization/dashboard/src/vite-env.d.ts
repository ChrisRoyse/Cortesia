/// <reference types="vite/client" />

interface ImportMetaEnv {
  VITE_WEBSOCKET_URL?: string
  VITE_API_BASE_URL?: string
  VITE_DEBUG?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}