What is a container?

A way to package application with all the necessary dependencies and configuration. Portable artifact, easily shared and moved around. Makes deployment more efficient.

Before containers, we had to install services on our OS directly. Installation process would be different on each OS environment.  There can also be dependency version conflicts. With containers, we do not have to directly install services. We have our own isolated environment packaged with all the needed configurations. We just need to fetch and run the docker.

Container is made of layers of images. Mostly linux base images, because they are small in size. Application image goes on top.

When the actual package is not running, it is docker image. If one actually starts the application, container environment is created, making it docker container.

Docker vs VM

OS have 2 layes: os kernel and applications. Docker virtualizes the application layer. VM virtualizes applications and os kernel. Docker image is much smaller because it only has one layer. Docker also runs faster as it does not have to boot os kernel. The only downside is that docker image may not be compatible with the os kernel.