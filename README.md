# quasi-detect
This is a project meant to automate detecting celestial objects. At the moment the project is not even functional yet and very little code is available, and it WILL take a couple of months. (of course it will be FULLY open source).

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

I've decided to use a ~~CNN~~ RNN to learn from spectra from SDSS. The reasoning for this is that , well, it just looked like there is MORE data for the model to process than with dim little dots that are like 4 pixels across if I used pictures (of course there are brighter ones, but most quasars are like this). I cannot use the entire catalog at the moment since my computer only has 8GB RAM and it pretty slow overall so I can't overuse batching unless its worth it to wait months for a simple file full of weights. Also, I will probably implement the PowerSign optimizer from [link to paper](https://arxiv.org/abs/1709.07417) to improve the time it takes to converge, which is very important with my very limited hardware. A custom loss is also in order.\0

## License

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see [this](https://www.gnu.org/licenses).

## How can you help

The most helpful thing would be contributing to the codebase. This is my first serious project ,so if you see bad code quality, please tell me.



