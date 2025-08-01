//! Sample Document Fixtures
//! 
//! Provides realistic document samples for testing knowledge processing
//! and hierarchical storage capabilities.

/// Simple document for basic processing tests
pub const SIMPLE_DOCUMENT: &str = r#"
The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
Foxes are carnivorous mammals belonging to the Canidae family. They are known for their intelligence and adaptability.
"#;

/// Complex scientific document for advanced processing tests
pub const COMPLEX_SCIENTIFIC_DOCUMENT: &str = r#"
# Quantum Computing and Machine Learning Integration

## Abstract

Quantum computing represents a paradigm shift in computational capabilities, offering exponential speedups for specific problem classes. Machine learning, particularly neural networks, has demonstrated remarkable success in pattern recognition and data analysis. The intersection of these fields promises revolutionary advances in artificial intelligence.

## Introduction

Quantum computers leverage quantum mechanical phenomena such as superposition and entanglement to process information. Unlike classical bits that exist in definite states of 0 or 1, quantum bits (qubits) can exist in superposition states, enabling parallel computation across multiple possibilities simultaneously.

Machine learning algorithms, especially deep neural networks, require extensive computational resources for training and inference. The computational complexity of training large language models scales exponentially with model size and dataset magnitude.

## Quantum Machine Learning Algorithms

### Variational Quantum Eigensolver (VQE)

The Variational Quantum Eigensolver represents a hybrid classical-quantum algorithm designed to find ground state energies of molecular Hamiltonians. VQE utilizes parameterized quantum circuits optimized through classical optimization techniques. The algorithm shows promise for drug discovery and materials science applications.

### Quantum Neural Networks

Quantum neural networks (QNNs) employ quantum circuits as computational units analogous to classical neurons. These networks can potentially achieve exponential speedups for specific learning tasks. However, current quantum hardware limitations restrict practical implementations to small-scale problems.

## Applications and Future Directions

Quantum-enhanced machine learning could revolutionize:
- Cryptography and security protocols
- Financial modeling and risk analysis  
- Drug discovery and molecular simulation
- Optimization problems in logistics
- Natural language processing for quantum chemistry

The National Institute of Standards and Technology (NIST) has established quantum computing standards to guide future development. IBM, Google, and Microsoft have invested billions in quantum research initiatives.

## Challenges and Limitations

Current quantum computers suffer from:
1. High error rates due to quantum decoherence
2. Limited qubit connectivity and coherence times
3. Scalability challenges for fault-tolerant systems
4. Need for specialized programming paradigms

## Conclusion

The convergence of quantum computing and machine learning represents one of the most promising frontiers in computational science. While significant technical hurdles remain, continued research and development may unlock unprecedented computational capabilities within the next decade.
"#;

/// Multi-topic document for relationship extraction tests
pub const MULTI_TOPIC_DOCUMENT: &str = r#"
# Technology Giants and Their AI Initiatives

## Google (Alphabet Inc.)

Google, founded by Larry Page and Sergey Brin at Stanford University in 1998, has become a leader in artificial intelligence research. The company's DeepMind subsidiary, acquired in 2014 for $500 million, developed AlphaGo, which defeated world champion Go player Lee Sedol in 2016.

Google's AI research focuses on natural language processing, computer vision, and reinforcement learning. The company's Transformer architecture, introduced in the paper "Attention Is All You Need" by Vaswani et al., revolutionized language modeling and enabled large language models like GPT and BERT.

## Microsoft Corporation  

Microsoft, established by Bill Gates and Paul Allen in 1975, has invested heavily in AI through its partnership with OpenAI. The company invested $1 billion in OpenAI in 2019 and an additional $10 billion in 2023, gaining exclusive access to GPT models for commercial applications.

Microsoft's Azure cloud platform provides AI services including cognitive services, machine learning pipelines, and GPU clusters for model training. The company's Copilot assistant integrates GPT-4 into Office applications, transforming productivity workflows.

## Meta (formerly Facebook)

Meta, founded by Mark Zuckerberg at Harvard University in 2004, focuses on AI for social media and virtual reality applications. The company's FAIR (Facebook AI Research) laboratory, led by Yann LeCun, conducts fundamental research in computer vision, natural language processing, and robotics.

Meta's LLaMA (Large Language Model Meta AI) family of models competes with OpenAI's GPT series. The company has open-sourced many AI models and tools, contributing to the broader research community.

## Competitive Dynamics

The AI race involves significant talent acquisition, with companies offering million-dollar compensation packages to attract top researchers. Google's brain drain to OpenAI in 2022 highlighted the competitive landscape for AI talent.

Patent portfolios in AI have become strategic assets. IBM holds over 9,000 AI-related patents, while Google and Microsoft have filed thousands of applications for machine learning innovations.

Research collaboration between academia and industry has accelerated AI progress. Stanford's Human-Centered AI Institute partners with multiple tech companies, while MIT's Computer Science and Artificial Intelligence Laboratory (CSAIL) collaborates on robotics and machine learning projects.
"#;

/// Document with temporal relationships for testing
pub const TEMPORAL_DOCUMENT: &str = r#"
# The Evolution of Artificial Intelligence

## Early Foundations (1940s-1950s)

In 1943, Warren McCulloch and Walter Pitts published their groundbreaking paper on artificial neurons, laying the mathematical foundation for neural networks. This work preceded the development of electronic computers by several years.

Alan Turing introduced the concept of machine intelligence in his 1950 paper "Computing Machinery and Intelligence," proposing what became known as the Turing Test. The same year, Claude Shannon published his information theory, providing the mathematical basis for digital communication.

## The Dartmouth Conference (1956)

John McCarthy organized the Dartmouth Summer Research Project on Artificial Intelligence in 1956, officially coining the term "artificial intelligence." The conference brought together researchers including Marvin Minsky, Allen Newell, and Herbert Simon, establishing AI as a distinct field of study.

Following the conference, significant funding flowed into AI research from organizations like DARPA (Defense Advanced Research Projects Agency), which was established in 1958 in response to the Soviet launch of Sputnik in 1957.

## Expert Systems Era (1970s-1980s)

The 1970s saw the rise of expert systems, beginning with DENDRAL in 1965 for chemical analysis, followed by MYCIN in 1976 for medical diagnosis. These systems captured human expertise in rule-based formats.

The first AI winter occurred from 1974 to 1980 due to over-promising and under-delivering on AI capabilities. Research funding decreased significantly during this period.

## Machine Learning Renaissance (1990s-2000s)

The development of backpropagation algorithm by Rumelhart, Hinton, and Williams in 1986 revitalized neural network research. This led to practical applications in the 1990s, including Yann LeCun's work on convolutional neural networks for handwritten digit recognition.

IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997, demonstrating AI's potential in strategic games. This victory came after Deep Blue's loss to Kasparov in 1996.

## Deep Learning Revolution (2010s-Present)

AlexNet's victory in the ImageNet competition in 2012 marked the beginning of the deep learning revolution. This breakthrough was enabled by GPU computing and large datasets.

Google's AlphaGo defeated world champion Lee Sedol in 2016, followed by AlphaZero in 2017, which learned chess, shogi, and Go from scratch. OpenAI's GPT-3 launch in 2020 demonstrated large language model capabilities, leading to ChatGPT's release in November 2022.

The current AI boom has attracted unprecedented investment, with companies like NVIDIA experiencing massive growth due to GPU demand for AI training.
"#;

/// Short document for testing edge cases
pub const SHORT_DOCUMENT: &str = "AI is transforming technology.";

/// Empty document for testing empty content handling
pub const EMPTY_DOCUMENT: &str = "";

/// Document with special characters and formatting
pub const FORMATTED_DOCUMENT: &str = r#"
# Special Characters & Formatting Test 📊

This document contains various special characters: αβγδε, 中文字符, العربية, 🚀🔬💡

**Bold text** and *italic text* should be handled correctly.

1. First item
2. Second item  
3. Third item

- Bullet point 1
- Bullet point 2
  - Nested bullet

> This is a blockquote with important information.

```python
def hello_world():
    print("Hello, World!")
    return True
```

Email: test@example.com
URL: https://www.example.com
Phone: (555) 123-4567

Mathematical notation: E = mc², ∑(x²), ∫f(x)dx

Currency symbols: $100, €85, ¥500, £75
"#;

/// Performance test document (large content)
pub const LARGE_DOCUMENT: &str = include_str!("large_test_document.txt");

/// Document samples for different complexity levels
pub struct DocumentSamples;

impl DocumentSamples {
    pub fn get_simple_samples() -> Vec<(&'static str, &'static str)> {
        vec![
            ("simple", SIMPLE_DOCUMENT),
            ("short", SHORT_DOCUMENT),
            ("empty", EMPTY_DOCUMENT),
        ]
    }
    
    pub fn get_medium_samples() -> Vec<(&'static str, &'static str)> {
        vec![
            ("formatted", FORMATTED_DOCUMENT),
            ("multi_topic", MULTI_TOPIC_DOCUMENT),
        ]
    }
    
    pub fn get_complex_samples() -> Vec<(&'static str, &'static str)> {
        vec![
            ("scientific", COMPLEX_SCIENTIFIC_DOCUMENT),
            ("temporal", TEMPORAL_DOCUMENT),
            ("large", LARGE_DOCUMENT),
        ]
    }
    
    pub fn get_all_samples() -> Vec<(&'static str, &'static str)> {
        let mut all = Vec::new();
        all.extend(Self::get_simple_samples());
        all.extend(Self::get_medium_samples());
        all.extend(Self::get_complex_samples());
        all
    }
}

/// Expected processing times for performance testing
pub struct ProcessingBenchmarks;

impl ProcessingBenchmarks {
    pub fn simple_document_max_time_ms() -> u64 { 100 }
    pub fn medium_document_max_time_ms() -> u64 { 500 }
    pub fn complex_document_max_time_ms() -> u64 { 2000 }
    pub fn large_document_max_time_ms() -> u64 { 10000 }
}