"""
Vacuum cleaner environment
Adrien Nivaggioli & Elie Khairallah
"""
from gym import spaces
import gym
import numpy as np
from scipy.stats import multivariate_normal
import random


class Roomba():
  def __init__(self, pos, direction, radius, battery, turn_angle=np.pi/9):
    self.pos = np.array(pos)
    self.direction = direction
    self.radius = radius
    self.turn_angle = turn_angle
    self.battery = battery
    self.max_battery = battery

  def set_pos(self, pos):
    self.pos = np.array(pos)

  def get_forward(self):
    new_x = np.sin(self.direction) + self.pos[0]
    new_y = np.cos(self.direction) + self.pos[1]
    return [new_x, new_y]

  def get_backward(self):
    new_x = np.sin(self.direction+np.pi) + self.pos[0]
    new_y = np.cos(self.direction+np.pi) + self.pos[1]
    return [new_x, new_y]

  def turn_right(self):
    self.direction = (self.direction - self.turn_angle) % (2*np.pi)


  def turn_left(self):
    self.direction = (self.direction + self.turn_angle) % (2*np.pi)

  def decrease_battery(self):
    self.battery -=1

  def recharge_battery(self):
    self.battery = self.max_battery

  def get_life(self):
    return self.battery / self.max_battery


  #https://stackoverflow.com/questions/49551440/python-all-points-on-circle-given-radius-and-center
  def get_cases_underneath(self, xmax, ymax, pos = None):
    if pos == None:
      pos = self.pos
    radius = self.radius
    x0 = pos[0] - 0.5
    y0 = pos[1] - 0.5

    L = []
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    for x, y in zip(x_[x], y_[y]):
      if x>=0 and y>=0 and x<xmax and y<ymax:
        L.append((x, y))
    return L


class RoombaEnv(gym.Env):

  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self, height, width, battery, roomba_radius=2):
    super(RoombaEnv, self).__init__()

    self.action_space = spaces.Discrete(4) #Forward, Backwards, Right, Left

    """                                      
    self.observation_space = spaces.Tuple([
        spaces.Box(low=0, high=4, shape=(height, width), dtype=np.float32), #1=cleanable, 2=wall, 3=charging station
        spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([height, width, 2*np.pi, 1]), dtype=np.float32)
    ])
    """
    #self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([height, width, 2*np.pi, 1]), dtype=np.float32)
    self.observation_space = spaces.Box(low=np.array(np.zeros(4+height*width)), high=np.concatenate([np.array([height, width, 2*np.pi, 1]), np.ones(height*width)]), dtype=np.float32)

    self.width = width
    self.height = height
    self.battery = battery
    self.roomba_radius = roomba_radius

    tmp = np.array( [[i,j] for i in range(height) for j in range(width)] )
    self.soil = 0.5 * multivariate_normal.pdf(tmp, mean=[height/2,width/2], cov=5*np.diag([1,1])).reshape((height, width)) #add cov

    self.room = np.ones((height, width))
    #obstacles = [[5,5], [5,6], [5,7], [5,8], [6,5], [7,5]]
    obstacles = []
    for obstIndex in obstacles:
      self.room[obstIndex[0],obstIndex[1]]= 2
    for i in range(4):
      for j in range(4):
        self.room[height-i-1, width-j-1] = 3
    self.viewer = None
    self.init_env()

  def init_env(self):
    self.roomba = Roomba( [self.height-3, self.width-3], -np.pi/2, self.roomba_radius, self.battery)
    self.done = False
    self.iteration = 0
    self.sum_rewards = 0
    self.dirty = np.ones((self.height, self.width))


  def clean_floor(self):
    cases_underneath = self.roomba.get_cases_underneath(self.height, self.width)
    #TODO: optimize
    reward = 0
    for case_underneath in cases_underneath:
      if self.room[case_underneath] == 1:
        reward += min(1,self.dirty[case_underneath])
        self.dirty[case_underneath] = 0

      if self.room[case_underneath] == 3:
        self.roomba.recharge_battery()
    return reward

  def step(self, action):
    self.iteration+=1
    if random.random() <0.1:
      action = random.randint(0,3)
    #current_reward = 1
    current_reward = 0
    if action == 0:
      new_pos = self.roomba.get_forward()
      if self.is_safe(new_pos):
        self.roomba.set_pos(new_pos)
      #else:
      #  current_reward += -10

    elif action == 1:
      new_pos = self.roomba.get_backward()
      if self.is_safe(new_pos):
        self.roomba.set_pos(new_pos)
      #else:
      #  current_reward += -10

    elif action == 2:
      self.roomba.turn_right()

    elif action == 3:
      self.roomba.turn_left()
    self.roomba.decrease_battery()
      
    current_reward += self.clean_floor()
    self.dirty = np.clip(self.soil + self.dirty, a_min = 0, a_max = 1) 

    if self.roomba.battery <= 0:
      current_reward += -100

    self.sum_rewards += current_reward

    if self.iteration>=10000 or self.roomba.battery <= 0:
      self.done = True
      
    return self.get_observation(),current_reward,self.done,{}

  def is_safe(self, new_pos):
    x = new_pos[0]
    y = new_pos[1]

    if y < 0 or x < 0 or x >= self.height or y >= self.width:
      return False

    cases_underneath = self.roomba.get_cases_underneath(self.height, self.width, new_pos)
    for case_underneath in cases_underneath:
      if self.room[case_underneath] == 2:
        return False

    return True

  def get_observation(self):
    #return np.array([self.roomba.pos[0]/self.height, self.roomba.pos[1]/self.width, self.roomba.direction, self.roomba.get_life()])
    return np.concatenate([np.array([self.roomba.pos[0]/self.height, self.roomba.pos[1]/self.width, self.roomba.direction, self.roomba.get_life()]), self.dirty.flatten()])

    #return (self.dirty, np.array([self.roomba.pos[0], self.roomba.pos[1], self.roomba.direction, self.roomba.get_life()]))

  def reset(self):
    self.init_env()
    return self.get_observation()

  def render(self, mode='human', close=False):
    scale = 6

    screen_width = scale*self.width
    screen_height = scale*self.height


    if self.viewer is None:
      #TODO: remove when it works
      import importlib
      import rendering
      importlib.reload(rendering)

      self.viewer = rendering.Viewer(screen_width, screen_height)

      self.cases = []
      for i in range(self.height):
        self.cases.append([])
        for j in range(self.width):
          x0 = i*scale
          y0 = j*scale
          x1 = (i+1)*scale
          y1 = (j+1)*scale
          case = rendering.FilledPolygon([(y0,x0), (y0,x1), (y1,x1), (y1,x0)])
          self.cases[i].append(case)
          self.viewer.add_geom(case)
      
      circle = rendering.make_circle(radius=self.roomba.radius*scale)
      circle.set_color(1,0,0)
      self.circle=circle
      direction_circle = rendering.make_circle(radius=self.roomba.radius*scale/2) #color?
      direction_circle.add_attr(rendering.Transform(translation=(self.roomba.radius*scale/2,0)))

      tracker = rendering.Compound([circle, direction_circle])

      self.tracker_trans = rendering.Transform()
      tracker.add_attr(self.tracker_trans)
      self.viewer.add_geom(tracker)

      
    for i in range(self.height):
        for j in range(self.width):
          if self.room[i,j] == 0:
            self.cases[i][j].set_color(1,1,1)
          elif self.room[i,j] == 1:
            self.cases[i][j].set_color(1 - 0.5 * min(1,self.dirty[i,j]),1 - 0.5 * min(1,self.dirty[i,j]),1 - 0.5 * min(1,self.dirty[i,j]))
          elif self.room[i,j] == 2:
            self.cases[i][j].set_color(0,0,0)
          elif self.room[i,j] == 3:
            self.cases[i][j].set_color(0.5,0.5,0)


    self.tracker_trans.set_translation(self.roomba.pos[1]*scale, self.roomba.pos[0]*scale)
    self.tracker_trans.set_rotation(self.roomba.direction)
    self.circle.set_color(self.roomba.get_life(),0,0)


    return self.viewer.render(return_rgb_array = mode=='rgb_array')
