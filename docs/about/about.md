# About and contact

**Axon SDK** is a neuromorphic framework to build spiking neural networks (SNN) for general-purpose computation.

**Axon SDK** was build by [Neucom](https://www.neucom.ai/). At Neucom, we're building a **general-purpose neuromorphic processor** that executes **deterministic computation** tasks with minimal energy consumption.

![Axon SDK logo](../figs/axon_logo.png)

**Axon SDK** is based on the theoretical work presented in [STICK](https://direct.mit.edu/neco/article-abstract/27/11/2261/8123/STICK-Spike-Time-Interval-Computational-Kernel-a?redirectedFrom=fulltext) and it expands it in several directions:

- **Axon SDK** provides a library of spiking computation kernels that can be combined to achieve complex computations.
- **Axon SDK** extends STICK with an arbitrary computation range, beyond the original constrain to scalars in the [0, 1] range.
- **Axon SDK** provides new computation primitives, such as a *Modulo* operation, *Scalar* multiplication and *Division*.
- **Axon SDK** provides a compiler that translates a user-defined algorithms using Python syntax syntax, into their spiking version.
- **Axon SDK** includes a *Simulator* to emulate the operation of the constructed SNN.

**Axon SDK** is open-sourced under a **GPLv3** license, preventing its inclusion in closed-source projects.

**Axon SDK** was developed by [Iñigo Lara](mailto:inigo@neucom.ai), [Francesco Sheiban](mailto:francesco@neucom.ai) and [Dmitri Lyalikov](mailto:dmitri@neucom.ai) and belongs to [Neucom APS](https://www.neucom.ai/). Neucom APS is based in Copenhagen, Denmark.

## Contact
If you’re working with Axon or STICK-based hardware and want to share your application, request features, or report issues, reach out via GitHub Issues or contact the Neucom team at `francesco@neucom.ai`.

![Neucom Logo](../figs/neucom_logo.png)
