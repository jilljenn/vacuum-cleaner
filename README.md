# Vacuum cleaner

## Install

It should be in Python 3. Please try `pip3` if `pip` does not work.

On Ubuntu you can make a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt  # Install gym and euclid3

Anywhere:

    pip install gym euclid3
    python cart.py  # To test the Gym environment
    python bounce.py  # Press A to randomly change the direction of the ball

## Useful resources

- [Environments in classic control or Box2D](https://gym.openai.com/envs/#classic_control)

Rendering involves either:

- reimplementing PolyLine etc. with pyglet, see [rendering.py](https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py)
- or OpenAI's Box2D fork, see [car_racing.py](https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py)

- Documentation of pyglet involving [Graphics](https://pyglet.readthedocs.io/en/stable/programming_guide/graphics.html)
- To make a circle you can also use OpenGL's [gluDisk](http://www.glprogramming.com/red/chapter11.html#name2)

Inspired by [bounce.py](https://gist.github.com/davepape/6993177) (come on, why did they use euclid3 instead of numpy?!)

- [How to create a new environment for Gym](https://github.com/openai/gym/blob/master/docs/creating-environments.md)
