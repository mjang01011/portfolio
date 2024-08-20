# Designing Machine Learning Systems

This is the page where I take my notes as I study with "Designing Machine Learning Systems" by Chip Huyen.

ML solutions generally do:

> Machine learning is an approach to (1) *learn* (2) *complex patterns* from (3) *existing data* and use these patterns to make (4) *predictions* on (5) *unseen data*.
>
> ![img](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098107956/files/assets/dmls_0102.png)

Figure 1-2. Instead of requiring hand-specified patterns to calculate outputs, ML solutions learn patterns from inputs and outputs

![image-20240810161636757](https://github.com/mjang01011/portfolio/blob/main/public/blogs/markdowns/images/image-20240810161636757.png?raw=true)

During model development, training is the bottleneck. Once the model has been deployed, however, its job is to generate predictions, so inference is the bottleneck. Research usually prioritizes fast training, whereas production usually prioritizes fast inference. Research prioritizes high throughput whereas production prioritizes low latency. In case you need a refresh, latency refers to the time it takes from receiving a query to returning the result. Throughput refers to how many queries are processed within a specific period of time. For example, the average latency of Google Translate is the average time it takes from when a user clicks Translate to when the translation is shown, and the throughput is how many queries it processes and serves a second.





The trend in the last decade shows that applications developed with the most/best data win. Instead of focusing on improving ML algorithms, most companies will focus on improving their data. Because data can change quickly, ML applications need to be adaptive to the changing environment, which might require faster development and deployment cycles.

Ch2

 A pattern I see in many short-lived ML projects is that the data scientists become too focused on hacking ML metrics without paying attention to business metrics.

systems should have these four characteristics: reliability, scalability, maintainability, and adaptability.

1. Reliability

   The system should continue to perform the correct function at the desired level of performance even in the face of adversity (hardware or software faults, and even human error). For example, if you use Google Translate to translate a sentence into a language you don’t know, it might be very hard for you to tell even if the translation is wrong

2. Scalability

   When talking about scalability most people think of resource scaling, which consists of up-scaling (expanding the resources to handle growth) and down-scaling (reducing the resources when not needed). For example, at peak, your system might require 100 GPUs (graphics processing units). However, most of the time, it needs only 10 GPUs. Keeping 100 GPUs up all the time can be costly, so your system should be able to scale down to 10 GPUs.

   An indispensable feature in many cloud services is autoscaling: automatically scaling up and down the number of machines depending on usage. This feature can be tricky to implement. Even Amazon fell victim to this when their autoscaling feature failed on Prime Day, causing their system to crash. An hour of downtime was estimated to cost Amazon between $72 million and $99 million.

3. Maintainability

   Code should be documented. Code, data, and artifacts should be versioned. Models should be sufficiently reproducible so that even when the original authors are not around, other contributors can have sufficient contexts to build on their work. When a problem occurs, different contributors should be able to work together to identify the problem and implement a solution without finger-pointing.

4. Adaptability

   Because ML systems are part code, part data, and data can change quickly, ML systems need to be able to evolve quickly. 



--

For example, here is one workflow that you might encounter when building an ML model to predict whether an ad should be shown when users enter a search query:[11](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch02.html#ch01fn45)

1. Choose a metric to optimize. For example, you might want to optimize for impressions—the number of times an ad is shown.

2. Collect data and obtain labels.

3. Engineer features.

4. Train models.

5. During error analysis, you realize that errors are caused by the wrong labels, so you relabel the data.

6. Train the model again.

7. During error analysis, you realize that your model always predicts that an ad shouldn’t be shown, and the reason is because 99.99% of the data you have have NEGATIVE labels (ads that shouldn’t be shown). So you have to collect more data of ads that should be shown.

8. Train the model again.

9. The model performs well on your existing test data, which is by now two months old. However, it performs poorly on the data from yesterday. Your model is now stale, so you need to update it on more recent data.

10. Train the model again.

11. Deploy the model.

12. The model seems to be performing well, but then the businesspeople come knocking on your door asking why the revenue is decreasing. It turns out the ads are being shown, but few people click on them. So you want to change your model to optimize for ad click-through rate instead.

13. Go to step 1.

    ![img](https://learning.oreilly.com/api/v2/epubs/urn:orm:book:9781098107956/files/assets/dmls_0202.png)

    

Step 1. Project scoping

A project starts with scoping the project, laying out goals, objectives, and constraints. Stakeholders should be identified and involved. Resources should be estimated and allocated. We already discussed different stakeholders and some of the foci for ML projects in production in [Chapter 1](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch01.html#overview_of_machine_learning_systems). We also already discussed how to scope an ML project in the context of a business earlier in this chapter. We’ll discuss how to organize teams to ensure the success of an ML project in [Chapter 11](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch11.html#the_human_side_of_machine_learning).

Step 2. Data engineering

A vast majority of ML models today learn from data, so developing ML models starts with engineering data. In [Chapter 3](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch03.html#data_engineering_fundamentals), we’ll discuss the fundamentals of data engineering, which covers handling data from different sources and formats. With access to raw data, we’ll want to curate training data out of it by sampling and generating labels, which is discussed in [Chapter 4](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch04.html#training_data).

Step 3. ML model development

With the initial set of training data, we’ll need to extract features and develop initial models leveraging these features. This is the stage that requires the most ML knowledge and is most often covered in ML courses. In [Chapter 5](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch05.html#feature_engineering), we’ll discuss feature engineering. In [Chapter 6](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch06.html#model_development_and_offline_evaluatio), we’ll discuss model selection, training, and evaluation.

Step 4. Deployment

After a model is developed, it needs to be made accessible to users. Developing an ML system is like writing—you will never reach the point when your system is done. But you do reach the point when you have to put your system out there. We’ll discuss different ways to deploy an ML model in [Chapter 7](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch07.html#model_deployment_and_prediction_service).

Step 5. Monitoring and continual learning

Once in production, models need to be monitored for performance decay and maintained to be adaptive to changing environments and changing requirements. This step will be discussed in Chapters [8](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch08.html#data_distribution_shifts_and_monitoring) and [9](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/ch09.html#continual_learning_and_test_in_producti).

Step 6. Business analysis

Model performance needs to be evaluated against business goals and analyzed to generate business insights. These insights can then be used to eliminate unproductive projects or scope out new projects. This step is closely related to the first step.



--

When the number of classes is high, such as disease diagnosis where the number of diseases can go up to thousands or product classifications where the number of products can go up to tens of thousands, we say the classification task has *high cardinality* (lots of unique #). High cardinality problems can be very challenging. The first challenge is in data collection. In my experience, ML models typically need at least 100 examples for each class to learn to classify that class. So if you have 1,000 classes, you already need at least 100,000 examples. The data collection can be especially difficult for rare classes. When you have thousands of classes, it’s likely that some of them are rare. When the number of classes is large, hierarchical classification might be useful. In hierarchical classification, you have a classifier to first classify each example into one of the large groups. Then you have another classifier to classify this example into one of the subgroups. For example, for product classification, you can first classify each product into one of the four main categories: electronics, home and kitchen, fashion, or pet supplies. After a product has been classified into a category, say fashion, you can use another classifier to put this product into one of the subgroups: shoes, shirts, jeans, or accessories.