# quasi-detect
This is a project meant to automate detecting celestial objects. At the moment the project is at a very early stage and the code looks embarassing, so it will take a while before I release any code (of course it will be FULLY open source).

## Tech Stack
<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/1920px-TensorFlow_logo.svg.png" alt="Tensorflow">
</div>

<div align="center">
  <img src="https://storage.googleapis.com/cms-storage-bucket/a9d6ce81aee44ae017ee.png" alt="Flutter">
</div>


## Motivation

1. Coding FUN!
2. Astronomy FUN!
3. Finding Deep Sky objects is a difficult job when done manually. While there are already approaches for automation, none of them have become really popular.
4. I want to see how well my idea works\0

## So WHAT is your approach then?

I've decided to use a CNN to learn from spectra from SDSS. The reasoning for this is that , well, it just looked like there is MORE data for the model to process than with dim little dots that are like 4 pixels across (of course there are brighter ones, but most quasars are like this). I cannot use the entire catalog at the moment since my computer only has 8GB RAM and it pretty slow overall so I can't overuse batching unless its worth it to wait months for a simple file full of weights. Also, I will probably implement the PowerSign optimizer from [link to paper](https://arxiv.org/abs/1709.07417) to improve the time it takes to converge, which is very important with my very limited hardware. A custom loss is also in order./

