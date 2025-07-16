# Fundamentals of Axon SDK

**Axon SDK** is a framework for easily creating and simulating spiking neural networks. It uses a custom neuron model that is useful for implementing several low-level computations that make it a Turing-complete computation framework. **Axon SDK** is the tool that allows the construction of spiking neural networks that execute user-defined computations.

**Axon SDK** includes several components that will be explored throughout the Getting Started and the Tutorials. Some of these are:

- A custom model of a *spiking neuron*.
- A *spiking network module*, defining connections between neurons.
- A *simulator*, for interfacing with the network and executing its operation.
- A *compiler*, for transforming user-defined computations into spiking networks

## Spiking neurons

Spiking neurons are the atomic concept of **Axon SDK**. A spiking neuron is a stateful component that will emit a spike when an internal buffer called *membrane potential* exceeds a threshold.
