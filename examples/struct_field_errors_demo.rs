// Example demonstrating E0063, E0560, and E0609 errors and their fixes

// Example struct definitions
#[derive(Debug)]
struct Person {
    name: String,
    age: u32,
    email: String,
}

#[derive(Debug)]
struct Company {
    name: String,
    employees: Vec<Person>,
    founded: u32,
}

// This would cause E0063: missing field in struct initializer
fn create_person_error() -> Person {
    Person {
        name: "John".to_string(),
        // Missing 'age' and 'email' fields - E0063
    }
}

// Fixed version
fn create_person_fixed() -> Person {
    Person {
        name: "John".to_string(),
        age: 30,
        email: "john@example.com".to_string(),
    }
}

// This would cause E0560: struct has no field named
fn create_company_error() -> Company {
    Company {
        name: "TechCorp".to_string(),
        employees: vec![],
        founded: 2020,
        // revenue: 1000000, // E0560 - no field named 'revenue'
    }
}

// This would cause E0609: no field on type
fn access_field_error(person: &Person) {
    // let salary = person.salary; // E0609 - no field 'salary' on type Person
    
    // Correct way:
    let name = &person.name;
    let age = person.age;
}

// Pattern for finding struct definitions and their usage
fn main() {
    let person = create_person_fixed();
    let company = Company {
        name: "Example Corp".to_string(),
        employees: vec![person],
        founded: 2024,
    };
    
    println!("Company: {:?}", company);
}

// Common patterns to search for when fixing these errors:
// 1. Struct definitions: struct StructName { fields }
// 2. Struct instantiation: StructName { field: value, ... }
// 3. Field access: instance.field_name
// 4. Pattern matching: match instance { StructName { field, .. } => ... }