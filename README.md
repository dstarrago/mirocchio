# MIRocchio Classification Algorithm
Prototype based classification algorithm for multiple-instance data

In contrasts to regular classification problems, in which each example has a unique description, in multiple-instance classification (MIC) problems, each example has many descriptions. In MIC, each example is called <em>bag</em>, and each description of a bag is an <em>instance</em>. In prototype based classification algorithm, a prototype is as an example that is computed for every class. When classifying a new example, the distance from the example to each prototype is calculated, and it is assigned to the class with the closer prototype. In the case of multiple-instance classification, the prototype is a bag. We used a variation of the Rocchio's formula to calculate the prototypes. This classification algorithm has proven to be useful for textual classification applications. More details can be found in 
- Sanchez Tarrago, D., Cornelis, C., Bello, R., Herrera, F.: A Multi-Instance Learning Wrapper Based on the Rocchio Classifier for Web Index Recommendation. Knowledge-Based Systems. 59, 173–181 (2014). <a href="https://www.sciencedirect.com/science/article/abs/pii/S0950705114000197" target="_blank">(text)</a>

Developed with:
- Java 1.8
- NetBeans IDE 8.2

Dependencies:
- Weka 3.7
- Weka package citationKNN 1.0.1  (included only for comparison purpose)
- Weka package multiInstanceLearning 1.0.10  (included only for comparison purpose)
- Weka package multiInstanceFilters 1.0.10  (included only for comparison purpose)
