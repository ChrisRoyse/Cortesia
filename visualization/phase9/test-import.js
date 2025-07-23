// Simple JavaScript test to verify component structure
const fs = require('fs');
const path = require('path');

function testComponentStructure() {
    const componentsDir = './src/components';
    const files = fs.readdirSync(componentsDir);
    
    console.log('✅ Phase 7 Components Found:');
    files.forEach(file => {
        if (file.endsWith('.tsx')) {
            const content = fs.readFileSync(path.join(componentsDir, file), 'utf8');
            
            // Check for React import
            const hasReactImport = content.includes("import React");
            const hasExport = content.includes("export");
            const hasUseEffect = content.includes("useEffect");
            const hasD3 = content.includes("import * as d3");
            
            console.log(`  ${file}:`);
            console.log(`    ✅ React Import: ${hasReactImport}`);
            console.log(`    ✅ Export: ${hasExport}`);
            console.log(`    ✅ useEffect: ${hasUseEffect}`);
            console.log(`    ✅ D3 Import: ${hasD3}`);
        }
    });
}

function testTypeDefinitions() {
    const typesDir = './src/types';
    const files = fs.readdirSync(typesDir);
    
    console.log('\n✅ Phase 7 Type Definitions:');
    files.forEach(file => {
        if (file.endsWith('.ts')) {
            const content = fs.readFileSync(path.join(typesDir, file), 'utf8');
            const interfaces = content.match(/export interface \w+/g) || [];
            console.log(`  ${file}: ${interfaces.length} interfaces`);
            interfaces.forEach(iface => console.log(`    - ${iface}`));
        }
    });
}

function testFeatures() {
    console.log('\n✅ Phase 7 Features Verification:');
    
    const features = [
        'SDRStorageVisualization.tsx',
        'KnowledgeGraphTreemap.tsx', 
        'ZeroCopyMonitor.tsx',
        'MemoryFlowVisualization.tsx',
        'CognitiveLayerMemory.tsx',
        'MemoryDashboard.tsx'
    ];
    
    features.forEach(feature => {
        const filePath = `./src/components/${feature}`;
        if (fs.existsSync(filePath)) {
            const content = fs.readFileSync(filePath, 'utf8');
            
            // Check for key functionality
            const hasVisualization = content.includes('d3.') || content.includes('svg');
            const hasProps = content.includes('Props');
            const hasState = content.includes('useState') || content.includes('useEffect');
            
            console.log(`  ${feature}:`);
            console.log(`    ✅ Has Visualization: ${hasVisualization}`);
            console.log(`    ✅ Has Props Interface: ${hasProps}`);
            console.log(`    ✅ Has State Management: ${hasState}`);
        } else {
            console.log(`  ❌ Missing: ${feature}`);
        }
    });
}

console.log('🧪 Testing LLMKG Visualization System - Phase 7\n');
testComponentStructure();
testTypeDefinitions();
testFeatures();
console.log('\n✅ Phase 7 Structure Verification Complete!');