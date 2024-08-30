Joshua Bloch - Effective Java

- Interface

`interface Exercise`

//Represent the cardio Exercise

public class cardioExerrcise implements Exercise{

}

(need to be public)

Stay away from Null

option<Exercise>



8/29

Class Diagrams

- Purpose: describe system organization and explain relationships between classes and interfaces

- UML is commonly used

Java Naming Convention

- Class names in TitleCase
- Interface names start with 'I'
- Field names in camelCase
- Primitives like int in lowercase

Systematic Method Design

1. Purpose question
2. Signature
   1. what is the method name and result type
3. Work through example
4. Translate to code
5. Test your code (Jacoco)
   1. division in int -> 0
   2. double floating point error
6. 