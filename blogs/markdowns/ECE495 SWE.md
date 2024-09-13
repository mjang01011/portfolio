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
   
   

9/3

public interface IATree {

​	// To return the yob of this class

​	Optional<Integer> getYearOfBirth()

}



Unknown: return Optional.empty()



Test:

@Test

void getYearOfBirth(){

​	# Can use assertAll, but dont know which test fails

​	assertAll(

​		()->assertEquals(Optional.of(2000), rachel.getYearOfBirth()),

​		()-> ...	

​	)

​	assertEquals(Optional.of(2000), rachel.getYearOfBirth());

​	assertEquals(Optional.empty(), unknown.getYearOfBirth())

}



HW:

system setIn, setOut functions for testing

java -jar CountWords_jar/CountWords.jar

vcm.duke.edu ubuntu system



9/5

git log -n1

git stash

ls -a (shows hidden files like .gitignore)

숙제: ubuntu VCM reserve 하고 거기서 테스트



DRY: Don't Repeat Yourself

- Using the same function for each class

  - Create an abstract "Exercise" class that brings up common functionalities

    `public abstract class`

    

    `public abstract class AExercise implements IExercise{`

    `public AExercise(){`

    `}`

    `public String getName(){`

    `}`

    `public String getCalories(){`

    `}`

    * We can just have a signature in abstract class as shown below

    `public abstract int duration()`

    `}`

- Use the same constant, and later realize you have to change the constant

- Define a variable for commonly used constants

- Define an own method for a repeated task that used across

// do I have enough time for this exercise (purpose)

`boolean enoughTime(int timeleft){ //template

​	`return timeleft > this.duration()` // hook, duration completed by the subclasses

}`

look at template hook design pattern

Why implement interface when we use abstract?



UML

circle / Cartpt topLeft, int radius, String color / double area(), boolean contain(Cartpt p)

square / Cartpt center, int size, String color / double area(), boolean contain(Cartpt p)



9/10

- Template Hook
- Interpreter Design Pattern
- Pair programming
- Constructors
- Constructor overloading



WM2: path from John to Mrie is #f, path from John to Bettina is [father, mother], path from John to John is []



9/12

Interface with fields:

`Interface Iconfig {`

​	`int DEFAULT_MAX = 200;`

`}`



illegal argument exception with meaningful error message



Java class Throwable



utils class for methods not directly related to classes such as date (ex. checkRange(min, max, message) that throws new exception)



How to unit test exceptions:

- assertions assertThrows, assertDoesNotThrow (when program should not throw error), when calling executables in the asserts, use lambda function for constructor ex. assertDoesNoetThrow(() -> {new Date(2024, 9, 12)})

throw new ???Exception("...message ...");
