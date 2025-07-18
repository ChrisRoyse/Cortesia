#[cfg(test)]
mod basic_tests {
    #[test]
    fn test_basic_math() {
        assert_eq!(2 + 2, 4);
        println!("✅ Basic math test passed!");
    }
    
    #[test]
    fn test_string_concat() {
        let s1 = "Hello";
        let s2 = "World";
        let result = format!("{} {}", s1, s2);
        assert_eq!(result, "Hello World");
        println!("✅ String concat test passed!");
    }
    
    #[test]
    fn test_vector_operations() {
        let mut vec = vec![1, 2, 3];
        vec.push(4);
        assert_eq!(vec.len(), 4);
        assert_eq!(vec[3], 4);
        println!("✅ Vector operations test passed!");
    }
}