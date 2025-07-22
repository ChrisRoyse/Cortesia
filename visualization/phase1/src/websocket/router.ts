/**
 * Message Routing and Subscription System
 * 
 * Handles topic-based message routing, subscription management,
 * and intelligent message distribution for WebSocket clients.
 */

import { EventEmitter } from 'events';
import { 
  WebSocketMessage, 
  MessageType, 
  DataTopic,
  SubscribeMessage,
  UnsubscribeMessage,
  SubscriptionAckMessage,
  ProtocolValidator,
  TopicManager
} from './protocol';
import { Logger } from '../utils/logger';

const logger = new Logger('MessageRouter');

export interface RouteFilter {
  topic?: string | RegExp;
  messageType?: MessageType | MessageType[];
  clientId?: string;
  custom?: (message: WebSocketMessage) => boolean;
}

export interface Subscription {
  id: string;
  clientId: string;
  topics: string[];
  filters?: Record<string, any>;
  createdAt: number;
  lastActivity: number;
  messageCount: number;
}

export interface RouteMatch {
  subscription: Subscription;
  message: WebSocketMessage;
  topic: string;
  matchedFilters: string[];
}

export class MessageRouter extends EventEmitter {
  private subscriptions = new Map<string, Subscription>();
  private clientSubscriptions = new Map<string, Set<string>>();
  private topicSubscriptions = new Map<string, Set<string>>();
  private messageCount = 0;
  private routingStats = {
    totalMessages: 0,
    totalRoutes: 0,
    filteredMessages: 0,
    subscriptionChanges: 0
  };

  constructor() {
    super();
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.on('subscription', (subscription: Subscription) => {
      this.handleSubscriptionChange(subscription);
    });

    this.on('unsubscription', (subscriptionId: string) => {
      this.handleUnsubscription(subscriptionId);
    });
  }

  /**
   * Route a message to all matching subscriptions
   */
  routeMessage(message: WebSocketMessage, topic?: string): RouteMatch[] {
    this.routingStats.totalMessages++;
    this.messageCount++;

    const matches: RouteMatch[] = [];
    const determinedTopic = topic || this.determineMessageTopic(message);

    if (!determinedTopic) {
      logger.warn('Unable to determine topic for message', { messageType: message.type });
      return matches;
    }

    // Find all subscriptions that match this topic
    const matchingSubscriptionIds = this.findMatchingSubscriptions(determinedTopic);

    for (const subscriptionId of matchingSubscriptionIds) {
      const subscription = this.subscriptions.get(subscriptionId);
      
      if (!subscription) {
        continue;
      }

      // Apply additional filters if any
      if (this.messagePassesFilters(message, subscription.filters)) {
        const match: RouteMatch = {
          subscription,
          message,
          topic: determinedTopic,
          matchedFilters: this.getMatchedFilters(message, subscription)
        };

        matches.push(match);
        
        // Update subscription activity
        subscription.lastActivity = Date.now();
        subscription.messageCount++;
      } else {
        this.routingStats.filteredMessages++;
      }
    }

    this.routingStats.totalRoutes += matches.length;

    logger.debug('Message routed', {
      messageType: message.type,
      topic: determinedTopic,
      matches: matches.length,
      messageId: message.id
    });

    return matches;
  }

  /**
   * Subscribe a client to topics
   */
  subscribe(subscribeMessage: SubscribeMessage): SubscriptionAckMessage {
    const { clientId, topics, filters } = subscribeMessage;
    
    // Validate topics
    const validTopics = topics.filter(topic => {
      if (TopicManager.isValidTopic(topic) || this.isWildcardTopic(topic)) {
        return true;
      }
      logger.warn('Invalid topic in subscription', { topic, clientId });
      return false;
    });

    if (validTopics.length === 0) {
      const errorAck: SubscriptionAckMessage = ProtocolValidator.createMessage(
        MessageType.SUBSCRIPTION_ACK,
        {
          topics,
          status: 'error',
          message: 'No valid topics provided'
        }
      );
      return errorAck;
    }

    // Create subscription
    const subscription: Subscription = {
      id: `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      clientId,
      topics: validTopics,
      filters,
      createdAt: Date.now(),
      lastActivity: Date.now(),
      messageCount: 0
    };

    // Store subscription
    this.subscriptions.set(subscription.id, subscription);
    
    // Update client subscriptions index
    if (!this.clientSubscriptions.has(clientId)) {
      this.clientSubscriptions.set(clientId, new Set());
    }
    this.clientSubscriptions.get(clientId)!.add(subscription.id);

    // Update topic subscriptions index
    for (const topic of validTopics) {
      if (!this.topicSubscriptions.has(topic)) {
        this.topicSubscriptions.set(topic, new Set());
      }
      this.topicSubscriptions.get(topic)!.add(subscription.id);
    }

    this.routingStats.subscriptionChanges++;
    this.emit('subscription', subscription);

    logger.info('Client subscribed to topics', { 
      clientId, 
      topics: validTopics,
      subscriptionId: subscription.id
    });

    const ack: SubscriptionAckMessage = ProtocolValidator.createMessage(
      MessageType.SUBSCRIPTION_ACK,
      {
        topics: validTopics,
        status: 'success',
        message: `Subscribed to ${validTopics.length} topics`
      }
    );

    return ack;
  }

  /**
   * Unsubscribe a client from topics
   */
  unsubscribe(unsubscribeMessage: UnsubscribeMessage): SubscriptionAckMessage {
    const { clientId, topics } = unsubscribeMessage;
    let unsubscribedTopics: string[] = [];

    const clientSubs = this.clientSubscriptions.get(clientId);
    if (!clientSubs) {
      const errorAck: SubscriptionAckMessage = ProtocolValidator.createMessage(
        MessageType.SUBSCRIPTION_ACK,
        {
          topics,
          status: 'error',
          message: 'No active subscriptions found for client'
        }
      );
      return errorAck;
    }

    // Remove specific topics or all if no topics specified
    for (const subscriptionId of Array.from(clientSubs)) {
      const subscription = this.subscriptions.get(subscriptionId);
      if (!subscription) continue;

      if (topics.length === 0) {
        // Unsubscribe from all topics
        unsubscribedTopics.push(...subscription.topics);
        this.removeSubscription(subscriptionId);
      } else {
        // Unsubscribe from specific topics
        const remainingTopics = subscription.topics.filter(topic => !topics.includes(topic));
        const removedTopics = subscription.topics.filter(topic => topics.includes(topic));
        
        if (remainingTopics.length === 0) {
          // Remove entire subscription if no topics remain
          this.removeSubscription(subscriptionId);
        } else {
          // Update subscription with remaining topics
          subscription.topics = remainingTopics;
          this.updateTopicIndices(subscriptionId, subscription.topics, removedTopics);
        }
        
        unsubscribedTopics.push(...removedTopics);
      }
    }

    this.routingStats.subscriptionChanges++;

    logger.info('Client unsubscribed from topics', { 
      clientId, 
      topics: unsubscribedTopics
    });

    const ack: SubscriptionAckMessage = ProtocolValidator.createMessage(
      MessageType.SUBSCRIPTION_ACK,
      {
        topics: unsubscribedTopics,
        status: 'success',
        message: `Unsubscribed from ${unsubscribedTopics.length} topics`
      }
    );

    return ack;
  }

  /**
   * Remove all subscriptions for a client
   */
  removeClientSubscriptions(clientId: string): void {
    const clientSubs = this.clientSubscriptions.get(clientId);
    if (!clientSubs) return;

    const subscriptionIds = Array.from(clientSubs);
    for (const subscriptionId of subscriptionIds) {
      this.removeSubscription(subscriptionId);
    }

    this.clientSubscriptions.delete(clientId);
    this.routingStats.subscriptionChanges++;

    logger.info('All subscriptions removed for client', { 
      clientId, 
      removedCount: subscriptionIds.length 
    });
  }

  /**
   * Get all subscriptions for a client
   */
  getClientSubscriptions(clientId: string): Subscription[] {
    const clientSubs = this.clientSubscriptions.get(clientId);
    if (!clientSubs) return [];

    const subscriptions: Subscription[] = [];
    for (const subscriptionId of clientSubs) {
      const subscription = this.subscriptions.get(subscriptionId);
      if (subscription) {
        subscriptions.push(subscription);
      }
    }

    return subscriptions;
  }

  /**
   * Get all subscriptions for a topic
   */
  getTopicSubscriptions(topic: string): Subscription[] {
    const topicSubs = this.topicSubscriptions.get(topic);
    if (!topicSubs) return [];

    const subscriptions: Subscription[] = [];
    for (const subscriptionId of topicSubs) {
      const subscription = this.subscriptions.get(subscriptionId);
      if (subscription) {
        subscriptions.push(subscription);
      }
    }

    return subscriptions;
  }

  /**
   * Determine the topic for a message based on its type and content
   */
  private determineMessageTopic(message: WebSocketMessage): string | null {
    switch (message.type) {
      case MessageType.COGNITIVE_PATTERN:
        return DataTopic.COGNITIVE_PATTERNS;
      
      case MessageType.NEURAL_ACTIVITY:
        return DataTopic.NEURAL_ACTIVITY;
      
      case MessageType.KNOWLEDGE_GRAPH_UPDATE:
        return DataTopic.KNOWLEDGE_GRAPH;
      
      case MessageType.SDR_OPERATION:
        return DataTopic.SDR_OPERATIONS;
      
      case MessageType.MEMORY_METRICS:
        return DataTopic.MEMORY_SYSTEM;
      
      case MessageType.ATTENTION_FOCUS:
        return DataTopic.ATTENTION_MECHANISM;
      
      case MessageType.TELEMETRY_DATA:
        return DataTopic.TELEMETRY;
      
      case MessageType.PERFORMANCE_METRICS:
        return DataTopic.PERFORMANCE;
      
      case MessageType.SYSTEM_STATUS:
      case MessageType.ERROR:
        return DataTopic.SYSTEM;
      
      default:
        return null;
    }
  }

  /**
   * Find subscriptions that match a given topic
   */
  private findMatchingSubscriptions(topic: string): Set<string> {
    const matchingIds = new Set<string>();

    // Direct topic matches
    const directMatches = this.topicSubscriptions.get(topic);
    if (directMatches) {
      for (const id of directMatches) {
        matchingIds.add(id);
      }
    }

    // Wildcard matches
    for (const [subscriptionTopic, subscriptionIds] of this.topicSubscriptions.entries()) {
      if (subscriptionTopic.includes('*') && TopicManager.matchesTopic(topic, subscriptionTopic)) {
        for (const id of subscriptionIds) {
          matchingIds.add(id);
        }
      }
    }

    return matchingIds;
  }

  /**
   * Check if a message passes the subscription filters
   */
  private messagePassesFilters(message: WebSocketMessage, filters?: Record<string, any>): boolean {
    if (!filters || Object.keys(filters).length === 0) {
      return true;
    }

    // Apply custom filter logic based on message type and filters
    for (const [filterKey, filterValue] of Object.entries(filters)) {
      switch (filterKey) {
        case 'messageTypes':
          if (Array.isArray(filterValue) && !filterValue.includes(message.type)) {
            return false;
          }
          break;
        
        case 'minConfidence':
          if ('data' in message && typeof message.data === 'object' && 
              'confidence' in message.data && typeof message.data.confidence === 'number') {
            if (message.data.confidence < filterValue) {
              return false;
            }
          }
          break;
        
        case 'patternTypes':
          if (message.type === MessageType.COGNITIVE_PATTERN && 
              'data' in message && typeof message.data === 'object' &&
              'patternType' in message.data) {
            if (Array.isArray(filterValue) && !filterValue.includes(message.data.patternType)) {
              return false;
            }
          }
          break;
        
        case 'minActivation':
          if ('data' in message && typeof message.data === 'object' && 
              'activation' in message.data && typeof message.data.activation === 'number') {
            if (message.data.activation < filterValue) {
              return false;
            }
          }
          break;
      }
    }

    return true;
  }

  /**
   * Get the filters that matched for a message and subscription
   */
  private getMatchedFilters(message: WebSocketMessage, subscription: Subscription): string[] {
    const matchedFilters: string[] = [];
    
    if (!subscription.filters) {
      return matchedFilters;
    }

    // Track which filters were applied
    for (const filterKey of Object.keys(subscription.filters)) {
      matchedFilters.push(filterKey);
    }

    return matchedFilters;
  }

  /**
   * Check if a topic is a wildcard pattern
   */
  private isWildcardTopic(topic: string): boolean {
    return topic.includes('*') || topic.includes('+');
  }

  /**
   * Remove a subscription by ID
   */
  private removeSubscription(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return;

    // Remove from subscriptions map
    this.subscriptions.delete(subscriptionId);

    // Remove from client subscriptions index
    const clientSubs = this.clientSubscriptions.get(subscription.clientId);
    if (clientSubs) {
      clientSubs.delete(subscriptionId);
      if (clientSubs.size === 0) {
        this.clientSubscriptions.delete(subscription.clientId);
      }
    }

    // Remove from topic subscriptions index
    for (const topic of subscription.topics) {
      const topicSubs = this.topicSubscriptions.get(topic);
      if (topicSubs) {
        topicSubs.delete(subscriptionId);
        if (topicSubs.size === 0) {
          this.topicSubscriptions.delete(topic);
        }
      }
    }

    this.emit('unsubscription', subscriptionId);
  }

  /**
   * Update topic indices when subscription topics change
   */
  private updateTopicIndices(subscriptionId: string, newTopics: string[], removedTopics: string[]): void {
    // Remove from old topics
    for (const topic of removedTopics) {
      const topicSubs = this.topicSubscriptions.get(topic);
      if (topicSubs) {
        topicSubs.delete(subscriptionId);
        if (topicSubs.size === 0) {
          this.topicSubscriptions.delete(topic);
        }
      }
    }

    // Add to new topics (if not already there)
    for (const topic of newTopics) {
      if (!this.topicSubscriptions.has(topic)) {
        this.topicSubscriptions.set(topic, new Set());
      }
      this.topicSubscriptions.get(topic)!.add(subscriptionId);
    }
  }

  /**
   * Handle subscription change events
   */
  private handleSubscriptionChange(subscription: Subscription): void {
    logger.debug('Subscription updated', {
      subscriptionId: subscription.id,
      clientId: subscription.clientId,
      topics: subscription.topics
    });
  }

  /**
   * Handle unsubscription events
   */
  private handleUnsubscription(subscriptionId: string): void {
    logger.debug('Subscription removed', { subscriptionId });
  }

  /**
   * Get routing statistics
   */
  getStats(): {
    totalSubscriptions: number;
    totalClients: number;
    totalTopics: number;
    routingStats: typeof this.routingStats;
    topSubscriptions: Array<{ topic: string; count: number }>;
  } {
    // Calculate top subscriptions by topic
    const topicCounts = new Map<string, number>();
    for (const [topic, subs] of this.topicSubscriptions.entries()) {
      topicCounts.set(topic, subs.size);
    }

    const topSubscriptions = Array.from(topicCounts.entries())
      .map(([topic, count]) => ({ topic, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    return {
      totalSubscriptions: this.subscriptions.size,
      totalClients: this.clientSubscriptions.size,
      totalTopics: this.topicSubscriptions.size,
      routingStats: this.routingStats,
      topSubscriptions
    };
  }

  /**
   * Clear all subscriptions and reset router
   */
  clear(): void {
    this.subscriptions.clear();
    this.clientSubscriptions.clear();
    this.topicSubscriptions.clear();
    this.messageCount = 0;
    this.routingStats = {
      totalMessages: 0,
      totalRoutes: 0,
      filteredMessages: 0,
      subscriptionChanges: 0
    };

    logger.info('Message router cleared');
  }
}

export { logger as routerLogger };