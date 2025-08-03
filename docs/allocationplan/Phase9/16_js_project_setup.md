# Micro-Phase 9.16: Setup TypeScript/JavaScript Project

## Objective
Initialize the JavaScript/TypeScript project structure for the CortexKG web interface with proper build configuration and type definitions.

## Prerequisites
- WASM module building successfully
- Node.js and npm installed
- Basic understanding of TypeScript configuration

## Task Description
Create the JavaScript wrapper project that will load and interface with the WASM module, providing a clean API for web developers.

## Specific Actions

1. **Create JavaScript project structure**:
   ```bash
   mkdir -p cortexkg-web
   cd cortexkg-web
   
   # Initialize package.json
   npm init -y
   
   # Create directory structure
   mkdir -p src/{core,ui,utils,types}
   mkdir -p public
   mkdir -p dist
   ```

2. **Install dependencies**:
   ```bash
   # TypeScript and build tools
   npm install --save-dev typescript @types/node webpack webpack-cli webpack-dev-server
   npm install --save-dev ts-loader html-webpack-plugin copy-webpack-plugin
   npm install --save-dev @types/jest jest ts-jest
   
   # Development dependencies
   npm install --save-dev prettier eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
   ```

3. **Create tsconfig.json**:
   ```json
   {
     "compilerOptions": {
       "target": "ES2020",
       "module": "ES2020",
       "lib": ["ES2020", "DOM", "DOM.Iterable"],
       "outDir": "./dist",
       "rootDir": "./src",
       "strict": true,
       "esModuleInterop": true,
       "skipLibCheck": true,
       "forceConsistentCasingInFileNames": true,
       "moduleResolution": "node",
       "resolveJsonModule": true,
       "declaration": true,
       "declarationMap": true,
       "sourceMap": true,
       "allowJs": true,
       "types": ["jest", "node"]
     },
     "include": ["src/**/*"],
     "exclude": ["node_modules", "dist", "**/*.test.ts"]
   }
   ```

4. **Create webpack configuration**:
   ```javascript
   // webpack.config.js
   const path = require('path');
   const HtmlWebpackPlugin = require('html-webpack-plugin');
   const CopyWebpackPlugin = require('copy-webpack-plugin');
   
   module.exports = {
     entry: './src/index.ts',
     output: {
       path: path.resolve(__dirname, 'dist'),
       filename: 'cortexkg.bundle.js',
       library: 'CortexKG',
       libraryTarget: 'umd',
       globalObject: 'this'
     },
     module: {
       rules: [
         {
           test: /\.tsx?$/,
           use: 'ts-loader',
           exclude: /node_modules/
         },
         {
           test: /\.css$/,
           use: ['style-loader', 'css-loader']
         }
       ]
     },
     resolve: {
       extensions: ['.tsx', '.ts', '.js', '.wasm']
     },
     experiments: {
       asyncWebAssembly: true,
       topLevelAwait: true
     },
     plugins: [
       new HtmlWebpackPlugin({
         template: './public/index.html',
         filename: 'index.html'
       }),
       new CopyWebpackPlugin({
         patterns: [
           { from: '../cortexkg-wasm/pkg', to: 'wasm' }
         ]
       })
     ],
     devServer: {
       static: {
         directory: path.join(__dirname, 'public')
       },
       compress: true,
       port: 8080,
       hot: true,
       headers: {
         'Cross-Origin-Embedder-Policy': 'require-corp',
         'Cross-Origin-Opener-Policy': 'same-origin'
       }
     }
   };
   ```

5. **Create type definitions**:
   ```typescript
   // src/types/cortexkg.d.ts
   declare module 'cortexkg-wasm' {
     export interface AllocationResult {
       readonly column_id: number;
       readonly confidence: number;
       readonly processing_time_ms: number;
     }
     
     export interface QueryResult {
       readonly concept_id: number;
       readonly content: string;
       readonly relevance_score: number;
       readonly activation_path: number[];
     }
     
     export interface PerformanceMetrics {
       readonly total_allocations: number;
       readonly average_allocation_time_ms: number;
       readonly memory_usage_bytes: number;
       readonly cache_hit_rate: number;
     }
     
     export interface CortexConfig {
       column_count?: number;
       max_connections_per_column?: number;
       enable_simd?: boolean;
       cache_size_mb?: number;
     }
     
     export class CortexKGWasm {
       constructor(config?: CortexConfig);
       initialize(): Promise<void>;
       initialize_with_storage(db_name: string): Promise<void>;
       allocate_concept(content: string): Promise<AllocationResult>;
       query(query_text: string): Promise<QueryResult[]>;
       get_performance_metrics(): PerformanceMetrics;
       get_column_states(): Array<{
         id: number;
         allocated: boolean;
         activation: number;
       }>;
       readonly is_initialized: boolean;
       readonly column_count: number;
     }
   }
   ```

6. **Create package.json scripts**:
   ```json
   {
     "scripts": {
       "build:wasm": "cd ../cortexkg-wasm && ./build.sh",
       "build:js": "webpack --mode production",
       "build": "npm run build:wasm && npm run build:js",
       "dev": "webpack serve --mode development",
       "test": "jest",
       "lint": "eslint src/**/*.ts",
       "format": "prettier --write src/**/*.ts",
       "typecheck": "tsc --noEmit"
     }
   }
   ```

7. **Create main entry point**:
   ```typescript
   // src/index.ts
   export { CortexKGWeb } from './core/CortexKGWeb';
   export { CorticalVisualizer } from './ui/CorticalVisualizer';
   export { QueryInterface } from './ui/QueryInterface';
   export { AllocationInterface } from './ui/AllocationInterface';
   export * from './types/cortexkg';
   
   // Auto-initialize if in browser environment
   if (typeof window !== 'undefined') {
     (window as any).CortexKG = {
       CortexKGWeb,
       CorticalVisualizer,
       QueryInterface,
       AllocationInterface
     };
   }
   ```

8. **Create ESLint configuration**:
   ```json
   // .eslintrc.json
   {
     "parser": "@typescript-eslint/parser",
     "extends": [
       "eslint:recommended",
       "plugin:@typescript-eslint/recommended"
     ],
     "parserOptions": {
       "ecmaVersion": 2020,
       "sourceType": "module"
     },
     "rules": {
       "@typescript-eslint/explicit-function-return-type": "warn",
       "@typescript-eslint/no-explicit-any": "warn",
       "@typescript-eslint/no-unused-vars": "error"
     }
   }
   ```

## Expected Outputs
- Complete TypeScript project structure
- Webpack configuration with WASM support
- Type definitions for WASM module
- Build and development scripts
- Linting and formatting configuration
- Entry point with auto-initialization

## Validation
1. Run `npm run typecheck` - no TypeScript errors
2. Run `npm run build` - builds successfully
3. Webpack dev server starts: `npm run dev`
4. Type definitions match WASM exports
5. ESLint runs without errors

## Next Steps
- Create WASM module loader (micro-phase 9.17)
- Build JavaScript API wrapper (micro-phase 9.18)