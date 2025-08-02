// High-performance string interning system to reduce memory usage for duplicate strings

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use ahash::AHashMap;

/// Interned string identifier - compact 32-bit reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InternedString(pub u32);

impl InternedString {
    pub const EMPTY: Self = InternedString(0);
    
    pub fn as_u32(self) -> u32 {
        self.0
    }
    
    pub fn from_u32(id: u32) -> Self {
        InternedString(id)
    }
}

impl Default for InternedString {
    fn default() -> Self {
        Self::EMPTY
    }
}

/// Thread-safe string interner with memory optimization
#[derive(Debug)]
pub struct StringInterner {
    // String storage - append-only for stability
    strings: RwLock<Vec<String>>,
    
    // Fast lookup from string to ID
    string_to_id: RwLock<AHashMap<String, InternedString>>,
    
    // Next available ID
    next_id: AtomicU32,
    
    // Memory usage tracking
    total_memory: AtomicU32,
    unique_strings: AtomicU32,
    total_references: AtomicU32,
}

impl StringInterner {
    pub fn new() -> Self {
        let interner = Self {
            strings: RwLock::new(Vec::new()),
            string_to_id: RwLock::new(AHashMap::new()),
            next_id: AtomicU32::new(1), // 0 reserved for empty
            total_memory: AtomicU32::new(0),
            unique_strings: AtomicU32::new(0),
            total_references: AtomicU32::new(0),
        };
        
        // Pre-intern empty string
        {
            let mut strings = interner.strings.write();
            let mut lookup = interner.string_to_id.write();
            
            strings.push(String::new());
            lookup.insert(String::new(), InternedString::EMPTY);
        }
        
        interner.unique_strings.store(1, Ordering::Relaxed);
        
        interner
    }
    
    /// Intern a string and return its ID
    pub fn intern<S: AsRef<str>>(&self, s: S) -> InternedString {
        let s = s.as_ref();
        
        // Fast path: check if already interned
        {
            let lookup = self.string_to_id.read();
            if let Some(&id) = lookup.get(s) {
                self.total_references.fetch_add(1, Ordering::Relaxed);
                return id;
            }
        }
        
        // Slow path: intern new string
        let id = InternedString(self.next_id.fetch_add(1, Ordering::Relaxed));
        
        {
            let mut strings = self.strings.write();
            let mut lookup = self.string_to_id.write();
            
            // Double-check in case another thread beat us
            if let Some(&existing_id) = lookup.get(s) {
                self.total_references.fetch_add(1, Ordering::Relaxed);
                return existing_id;
            }
            
            // Add new string
            strings.push(s.to_string());
            lookup.insert(s.to_string(), id);
            
            // Update statistics
            self.total_memory.fetch_add(s.len() as u32, Ordering::Relaxed);
            self.unique_strings.fetch_add(1, Ordering::Relaxed);
            self.total_references.fetch_add(1, Ordering::Relaxed);
        }
        
        id
    }
    
    /// Batch intern multiple strings efficiently
    pub fn intern_batch<S: AsRef<str>>(&self, strings: &[S]) -> Vec<InternedString> {
        strings.iter().map(|s| self.intern(s)).collect()
    }
    
    /// Get string by ID
    pub fn get(&self, id: InternedString) -> Option<String> {
        let strings = self.strings.read();
        strings.get(id.0 as usize).cloned()
    }
    
    
    /// Get statistics about memory usage
    pub fn stats(&self) -> InternerStats {
        InternerStats {
            unique_strings: self.unique_strings.load(Ordering::Relaxed),
            total_references: self.total_references.load(Ordering::Relaxed),
            total_memory_bytes: self.total_memory.load(Ordering::Relaxed),
            deduplication_ratio: {
                let refs = self.total_references.load(Ordering::Relaxed);
                let unique = self.unique_strings.load(Ordering::Relaxed);
                if unique > 0 { refs as f32 / unique as f32 } else { 1.0 }
            },
            memory_saved_bytes: {
                let refs = self.total_references.load(Ordering::Relaxed);
                let unique = self.unique_strings.load(Ordering::Relaxed);
                let total_mem = self.total_memory.load(Ordering::Relaxed);
                if refs > unique {
                    total_mem * (refs - unique) / refs
                } else {
                    0
                }
            }
        }
    }
    
    /// Clear all interned strings (preserves empty string)
    pub fn clear(&self) {
        let mut strings = self.strings.write();
        let mut lookup = self.string_to_id.write();
        
        strings.clear();
        lookup.clear();
        
        // Restore empty string
        strings.push(String::new());
        lookup.insert(String::new(), InternedString::EMPTY);
        
        self.next_id.store(1, Ordering::Relaxed);
        self.total_memory.store(0, Ordering::Relaxed);
        self.unique_strings.store(1, Ordering::Relaxed);
        self.total_references.store(0, Ordering::Relaxed);
    }
    
    /// Get all interned strings (for serialization)
    pub fn get_all_strings(&self) -> Vec<String> {
        let strings = self.strings.read();
        strings.clone()
    }
    
    /// Restore from serialized strings
    pub fn restore_from_strings(&self, strings: Vec<String>) {
        let mut strings_guard = self.strings.write();
        let mut lookup = self.string_to_id.write();
        
        strings_guard.clear();
        lookup.clear();
        
        let mut total_memory = 0;
        
        for (i, string) in strings.into_iter().enumerate() {
            total_memory += string.len();
            let id = InternedString(i as u32);
            lookup.insert(string.clone(), id);
            strings_guard.push(string);
        }
        
        self.next_id.store(strings_guard.len() as u32, Ordering::Relaxed);
        self.total_memory.store(total_memory as u32, Ordering::Relaxed);
        self.unique_strings.store(strings_guard.len() as u32, Ordering::Relaxed);
        self.total_references.store(0, Ordering::Relaxed); // Reset reference count
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct InternerStats {
    pub unique_strings: u32,
    pub total_references: u32,
    pub total_memory_bytes: u32,
    pub deduplication_ratio: f32,
    pub memory_saved_bytes: u32,
}

impl std::fmt::Display for InternerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "String Interner Stats:")?;
        writeln!(f, "  Unique strings: {}", self.unique_strings)?;
        writeln!(f, "  Total references: {}", self.total_references)?;
        writeln!(f, "  Memory used: {} bytes", self.total_memory_bytes)?;
        writeln!(f, "  Deduplication ratio: {:.1}:1", self.deduplication_ratio)?;
        write!(f, "  Memory saved: {} bytes", self.memory_saved_bytes)
    }
}

/// Interned property container for entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternedProperties {
    // Property name -> value mapping using interned strings
    properties: HashMap<InternedString, InternedString>,
}

impl InternedProperties {
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
        }
    }
    
    /// Add property with string interning
    pub fn insert(&mut self, interner: &StringInterner, key: &str, value: &str) {
        let key_id = interner.intern(key);
        let value_id = interner.intern(value);
        self.properties.insert(key_id, value_id);
    }
    
    /// Get property value by key
    pub fn get(&self, interner: &StringInterner, key: &str) -> Option<String> {
        let key_id = interner.intern(key);
        let value_id = self.properties.get(&key_id)?;
        interner.get(*value_id)
    }
    
    /// Iterate over all properties
    pub fn iter<'a>(&'a self, interner: &'a StringInterner) -> impl Iterator<Item = (String, String)> + 'a {
        self.properties.iter().filter_map(move |(k, v)| {
            let key = interner.get(*k)?;
            let value = interner.get(*v)?;
            Some((key, value))
        })
    }
    
    /// Get raw property mapping
    pub fn raw_properties(&self) -> &HashMap<InternedString, InternedString> {
        &self.properties
    }
    
    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.properties.len() * (std::mem::size_of::<InternedString>() * 2)
    }
    
    /// Number of properties
    pub fn len(&self) -> usize {
        self.properties.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }
    
    /// Create from key-value pairs
    pub fn from_pairs(interner: &StringInterner, pairs: &[(&str, &str)]) -> Self {
        let mut props = Self::new();
        for (key, value) in pairs {
            props.insert(interner, key, value);
        }
        props
    }
    
    /// Serialize to JSON string using interner
    pub fn to_json(&self, interner: &StringInterner) -> Result<String, serde_json::Error> {
        let map: HashMap<String, String> = self.iter(interner).collect();
        serde_json::to_string(&map)
    }
    
    /// Deserialize from JSON string using interner
    pub fn from_json(interner: &StringInterner, json: &str) -> Result<Self, serde_json::Error> {
        let map: HashMap<String, String> = serde_json::from_str(json)?;
        let mut props = Self::new();
        for (key, value) in map {
            props.insert(interner, &key, &value);
        }
        Ok(props)
    }
}

impl Default for InternedProperties {
    fn default() -> Self {
        Self::new()
    }
}

/// Global string interner for entity properties
pub struct GlobalStringInterner {
    interner: StringInterner,
}

impl Default for GlobalStringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalStringInterner {
    pub fn new() -> Self {
        Self {
            interner: StringInterner::new(),
        }
    }
    
    pub fn get() -> &'static GlobalStringInterner {
        static INSTANCE: std::sync::OnceLock<GlobalStringInterner> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(GlobalStringInterner::new)
    }
    
    pub fn intern<S: AsRef<str>>(&self, s: S) -> InternedString {
        self.interner.intern(s)
    }
    
    pub fn resolve(&self, id: InternedString) -> Option<String> {
        self.interner.get(id)
    }
    
    pub fn stats(&self) -> InternerStats {
        self.interner.stats()
    }
    
    pub fn clear(&self) {
        self.interner.clear()
    }
}

/// Convenience functions for global interner
pub fn intern_string<S: AsRef<str>>(s: S) -> InternedString {
    GlobalStringInterner::get().intern(s)
}

pub fn get_string(id: InternedString) -> Option<String> {
    GlobalStringInterner::get().resolve(id)
}

pub fn interner_stats() -> InternerStats {
    GlobalStringInterner::get().stats()
}

pub fn clear_interner() {
    GlobalStringInterner::get().clear()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_interning() {
        let interner = StringInterner::new();
        
        let id1 = interner.intern("hello");
        let id2 = interner.intern("world");
        let id3 = interner.intern("hello"); // Should reuse id1
        
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        
        assert_eq!(interner.get(id1), Some("hello".to_string()));
        assert_eq!(interner.get(id2), Some("world".to_string()));
    }
    
    #[test]
    fn test_properties() {
        let interner = StringInterner::new();
        let mut props = InternedProperties::new();
        
        props.insert(&interner, "name", "John");
        props.insert(&interner, "age", "30");
        
        assert_eq!(props.get(&interner, "name"), Some("John".to_string()));
        assert_eq!(props.get(&interner, "age"), Some("30".to_string()));
        assert_eq!(props.get(&interner, "missing"), None);
    }
    
    #[test]
    fn test_memory_savings() {
        let interner = StringInterner::new();
        
        // Intern the same string multiple times
        for _ in 0..100 {
            interner.intern("repeated_string");
        }
        
        let stats = interner.stats();
        assert_eq!(stats.unique_strings, 2); // empty + "repeated_string"
        assert_eq!(stats.total_references, 100);
        assert!(stats.deduplication_ratio > 1.0);
        assert!(stats.memory_saved_bytes > 0);
    }
}