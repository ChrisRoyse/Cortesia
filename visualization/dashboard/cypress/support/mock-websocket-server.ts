// Mock WebSocket Server for Testing

interface MockMessage {
  type: string
  data: any
  timestamp?: number
}

interface MockWebSocketServer {
  port: number
  clients: Set<MockWebSocketClient>
  isRunning: boolean
  messageQueue: MockMessage[]
  
  start(): void
  stop(): void
  broadcast(message: MockMessage): void
  sendToClient(clientId: string, message: MockMessage): void
}

interface MockWebSocketClient {
  id: string
  readyState: number
  send(data: string): void
  close(): void
  onMessage?: (data: string) => void
  onClose?: () => void
  onError?: (error: any) => void
}

class MockWebSocketServerImpl implements MockWebSocketServer {
  port: number
  clients: Set<MockWebSocketClient>
  isRunning: boolean
  messageQueue: MockMessage[]
  private messageInterval?: NodeJS.Timeout

  constructor(port: number = 9001) {
    this.port = port
    this.clients = new Set()
    this.isRunning = false
    this.messageQueue = []
  }

  start(): void {
    if (this.isRunning) return
    
    this.isRunning = true
    console.log(`Mock WebSocket server started on port ${this.port}`)
    
    // Simulate periodic brain updates
    this.messageInterval = setInterval(() => {
      this.sendBrainUpdate()
    }, 1000)
  }

  stop(): void {
    if (!this.isRunning) return
    
    this.isRunning = false
    
    if (this.messageInterval) {
      clearInterval(this.messageInterval)
    }
    
    // Close all client connections
    this.clients.forEach(client => client.close())
    this.clients.clear()
    
    console.log('Mock WebSocket server stopped')
  }

  broadcast(message: MockMessage): void {
    const messageStr = JSON.stringify({
      ...message,
      timestamp: message.timestamp || Date.now()
    })
    
    this.clients.forEach(client => {
      if (client.readyState === 1) { // OPEN
        client.send(messageStr)
      }
    })
  }

  sendToClient(clientId: string, message: MockMessage): void {
    const client = Array.from(this.clients).find(c => c.id === clientId)
    if (client && client.readyState === 1) {
      const messageStr = JSON.stringify({
        ...message,
        timestamp: message.timestamp || Date.now()
      })
      client.send(messageStr)
    }
  }

  addClient(client: MockWebSocketClient): void {
    this.clients.add(client)
    console.log(`Client ${client.id} connected. Total clients: ${this.clients.size}`)
  }

  removeClient(client: MockWebSocketClient): void {
    this.clients.delete(client)
    console.log(`Client ${client.id} disconnected. Total clients: ${this.clients.size}`)
  }

  private sendBrainUpdate(): void {
    if (this.clients.size === 0) return

    const brainUpdate = {
      type: 'brain_update',
      data: {
        entities: this.generateMockEntities(50),
        relationships: this.generateMockRelationships(100),
        statistics: this.generateMockStatistics(),
        cognitive_patterns: this.generateMockCognitivePatterns(),
        memory_stats: this.generateMockMemoryStats()
      }
    }

    this.broadcast(brainUpdate)
  }

  private generateMockEntities(count: number) {
    return Array.from({ length: count }, (_, i) => ({
      id: `entity_${i}`,
      type_id: (i % 4) + 1,
      activation: Math.random(),
      direction: ['Input', 'Output', 'Hidden', 'Gate'][i % 4],
      properties: {
        name: `Entity ${i}`,
        created_at: new Date().toISOString()
      },
      embedding: Array.from({ length: 128 }, () => Math.random() * 2 - 1)
    }))
  }

  private generateMockRelationships(count: number) {
    return Array.from({ length: count }, (_, i) => ({
      id: `rel_${i}`,
      from: `entity_${Math.floor(Math.random() * 50)}`,
      to: `entity_${Math.floor(Math.random() * 50)}`,
      weight: Math.random(),
      inhibitory: Math.random() > 0.9,
      rel_type: Math.floor(Math.random() * 4) + 1,
      hebbian_strength: Math.random(),
      last_activation: Date.now() - Math.random() * 10000
    }))
  }

  private generateMockStatistics() {
    return {
      total_entities: 50,
      total_relationships: 100,
      avg_activation: 0.45 + Math.random() * 0.1,
      max_activation: 0.8 + Math.random() * 0.2,
      min_activation: Math.random() * 0.2,
      active_entities: Math.floor(Math.random() * 20) + 10,
      learning_rate: 0.01 + Math.random() * 0.02,
      memory_consolidation_rate: Math.random() * 0.05
    }
  }

  private generateMockCognitivePatterns() {
    const patterns = [
      'convergent_thinking',
      'divergent_thinking', 
      'lateral_thinking',
      'systems_thinking',
      'critical_thinking',
      'abstract_thinking',
      'adaptive_thinking'
    ]

    return patterns.map(pattern => ({
      type: pattern,
      activation_level: Math.random(),
      confidence: Math.random(),
      pattern_strength: Math.random(),
      last_triggered: Date.now() - Math.random() * 5000
    }))
  }

  private generateMockMemoryStats() {
    return {
      working_memory_usage: Math.random() * 0.8,
      long_term_consolidation_rate: Math.random() * 0.3,
      sdr_compression_ratio: 0.1 + Math.random() * 0.1,
      total_memories: Math.floor(Math.random() * 1000) + 500,
      recent_memories: Math.floor(Math.random() * 50) + 20,
      memory_retrieval_accuracy: 0.8 + Math.random() * 0.2
    }
  }

  // Simulate specific test scenarios
  simulateConnectionError(): void {
    this.broadcast({
      type: 'connection_error',
      data: {
        error: 'Connection lost',
        code: 1006,
        reason: 'Abnormal closure'
      }
    })
  }

  simulateHighLoad(): void {
    // Send multiple rapid updates
    for (let i = 0; i < 10; i++) {
      setTimeout(() => {
        this.sendBrainUpdate()
      }, i * 100)
    }
  }

  simulateMalformedMessage(): void {
    this.clients.forEach(client => {
      if (client.readyState === 1) {
        client.send('{ "invalid": json, "missing": quote }')
      }
    })
  }

  simulateLargeMessage(): void {
    const largeData = Array.from({ length: 10000 }, (_, i) => ({
      id: `large_entity_${i}`,
      data: 'x'.repeat(1000) // Large data payload
    }))

    this.broadcast({
      type: 'large_data_update',
      data: largeData
    })
  }
}

// Create global mock server instance
let mockServer: MockWebSocketServerImpl | null = null

// Commands for Cypress to control the mock server
Cypress.Commands.add('startMockWebSocketServer', (port?: number) => {
  if (mockServer && mockServer.isRunning) {
    mockServer.stop()
  }
  
  mockServer = new MockWebSocketServerImpl(port || 9001)
  mockServer.start()
  
  return cy.wrap({
    port: mockServer.port,
    status: 'running'
  })
})

Cypress.Commands.add('stopMockWebSocketServer', () => {
  if (mockServer) {
    mockServer.stop()
    mockServer = null
  }
  
  return cy.wrap({
    status: 'stopped'
  })
})

Cypress.Commands.add('mockWebSocketBroadcast', (message: MockMessage) => {
  if (mockServer && mockServer.isRunning) {
    mockServer.broadcast(message)
  } else {
    throw new Error('Mock WebSocket server is not running')
  }
})

Cypress.Commands.add('simulateWebSocketScenario', (scenario: string) => {
  if (!mockServer || !mockServer.isRunning) {
    throw new Error('Mock WebSocket server is not running')
  }

  switch (scenario) {
    case 'connection_error':
      mockServer.simulateConnectionError()
      break
    case 'high_load':
      mockServer.simulateHighLoad()
      break
    case 'malformed_message':
      mockServer.simulateMalformedMessage()
      break
    case 'large_message':
      mockServer.simulateLargeMessage()
      break
    default:
      throw new Error(`Unknown scenario: ${scenario}`)
  }
})

declare global {
  namespace Cypress {
    interface Chainable {
      startMockWebSocketServer(port?: number): Chainable<{ port: number, status: string }>
      stopMockWebSocketServer(): Chainable<{ status: string }>
      mockWebSocketBroadcast(message: MockMessage): Chainable<void>
      simulateWebSocketScenario(scenario: string): Chainable<void>
    }
  }
}

export { MockWebSocketServerImpl, type MockMessage, type MockWebSocketServer, type MockWebSocketClient }