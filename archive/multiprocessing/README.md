# Using Multiprocessing for pre- or post-processing steps

Truss server executes the steps `preprocssing`, `predict` and `postprocess` (or a sub set) using python asyncio. This allows concurrency - and therefore higher throughput - while waiting for I/O-bound tasks (if asyncio-supporting libraries are used e.g. fetching data with async HTTP requrests).
However, it does not increase throughput, if the work in any of those steps is CPU-bound (either by blocking I/O functions or compute intense pre/post-processing operations).

Specifically, if pre/post-processing is CPU-bound, it can lead to underutilization of the server's GPU resources and suboptimal throughput.
In this case, simply increasing the number server replicas is not efficient, because it would likewise increase the GPU footprint.

A possible solution is to move the CPU work into separate processes that do not block the main server from using the GPU for *different* requests in the meantime.

This dummy model illustrates such a setup. To adopt it, the overall inference pipeline must  be structured such that most of the CPU work goes into the pre- or post-processing methods and the `predict` method contains mostly GPU work.


## Comments on the example code:
* Multiprocessing only demonstrated for `preprocessing`, for `postprocessing` the same pattern (with same process pool) can be used.
* Multiprocessing depends on pickling the function to run (and all its context) to the process pool. If the function depends on unpickleable objects or "large" objects this can be problematic and a different design is needed (see comments in code).
* The worker pool size is determined by the number of CPUs visible to the server.
