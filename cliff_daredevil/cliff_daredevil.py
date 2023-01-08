from typing import Optional, Callable
import Box2D as b2
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .car_model import CarModel, CAR_WIDTH, CAR_HEIGHT

ROAD_HIGHT = 25.0
DT = 1 / 60


class FrictionZoneListener(b2.b2ContactListener):
    def __init__(self, env):
        b2.b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        if (
            contact.fixtureA == self.env.friction_zone
            or contact.fixtureB == self.env.friction_zone
        ):
            if begin:
                self.env.friction_zone.touch = True
            else:
                self.env.friction_zone.touch = False


class CliffDaredevil(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self, render_mode: Optional[str] = None, friction_profile: float = 0.1
    ):
        self.contactListener_keepref = FrictionZoneListener(self)
        self.min_position = -5.0
        self.max_position = 75.0
        self.goal_zone = (50.0, 50.0 + CAR_WIDTH)
        self.cliff_edge = 0.3
        self.friction_start = 40.0
        self.friction = (
            friction_profile
            if callable(friction_profile)
            else lambda _: friction_profile
        )
        self.world = b2.b2World((0, -10), contactListener=self.contactListener_keepref)
        self._build_road()
        self.viewer = None
        self.action_space = spaces.Box(
            np.array([-1, 0]), np.array([+1, +1]), dtype=np.float32
        )  # gas, brake
        self.observation_space = spaces.Box(
            low=np.array([self.min_position, -32.0]),
            high=np.array([self.max_position, 32.0]),
            dtype=np.float32,
        )  # position, velocity
        self.car: Optional[CarModel] = None
        self.render_mode = render_mode
        self.seed()
        self.reset()

    def _build_road(self):
        self.ground = self.world.CreateStaticBody(
            fixtures=[
                b2.b2FixtureDef(
                    shape=b2.b2EdgeShape(
                        vertices=[
                            (self.min_position, ROAD_HIGHT),
                            (self.goal_zone[1] + self.cliff_edge, ROAD_HIGHT),
                        ]
                    ),
                    friction=0.99,
                ),
                b2.b2FixtureDef(
                    shape=b2.b2EdgeShape(
                        vertices=[
                            (self.friction_start, ROAD_HIGHT),
                            (self.goal_zone[1], ROAD_HIGHT),
                        ]
                    ),
                    friction=self.friction(0),
                ),
            ]
        )
        self.ground.fixtures[1].touch = False
        self.friction_zone = self.ground.fixtures[1]
        self.ground.userData = self.ground

    # In new Gym API, this function is deprecated
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        truncated = False
        if action is not None:
            action = np.clip(action, -1.0, 1.0)
            self.car.gas(action[0])
            self.car.brake(action[1])
        self.car.step()
        self.world.Step(DT, 6 * 30, 2 * 30)
        x, y = self.car.hull.position
        if self.friction_zone.touch:
            progress = np.clip(
                (x - self.friction_start) / (self.goal_zone[1] - self.friction_start),
                0.0,
                1.0,
            )
            friction = self.friction(progress)
            self.friction_zone.friction = friction
        backward = x < self.min_position
        reward = 0
        if backward:
            reward -= 1.0
        angle = self.car.hull.angle
        terminated = y < ROAD_HIGHT or backward or np.abs(angle) > 0.9
        if self.goal_zone[0] < x < self.goal_zone[1]:
            reward += 1.0
        distance = self.goal_zone[0] - x
        reward -= np.abs(distance)
        cost = -(self.goal_zone[1] + self.cliff_edge - x)
        v = self.car.hull.linearVelocity[0]
        if self.render_mode == "human":
            self.render()
        return (
            np.array([x, v], np.float32),
            reward,
            terminated,
            truncated,
            {"cost": cost},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self._destroy()
        self.friction_zone.touch = False
        #assert type(self.friction) is Callable
        self.friction_zone.friction = self.friction(0.0)
        position = self.np_random.uniform(low=-0.1, high=0.1)
        self.car = CarModel(self.world, position, ROAD_HIGHT)
        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _destroy(self):
        if self.car is None:
            return
        self.car.destroy()

    def render(self):
        mode = self.render_mode
        screen_width, screen_height = 640, 320
        if self.viewer is None:
            import cliff_daredevil.rendering as rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(self.min_position, self.max_position, 0.0, 40.0)
            sky = rendering.make_polygon(
                [
                    (self.min_position, 0.0),
                    (self.min_position, 40.0),
                    (self.max_position, 40.0),
                    (self.max_position, 0.0),
                ]
            )
            sky.set_color(135 / 255, 206 / 255, 235 / 255)
            xs_ground = np.linspace(self.goal_zone[1], self.goal_zone[1] + 1.0, 25)
            ys_ground = np.linspace(ROAD_HIGHT, 0.0, 25)
            xs_ground += self.np_random.uniform(-0.75, 0.75, 25)
            ground = rendering.make_polygon(
                [(self.min_position, 0.0), (self.min_position, ROAD_HIGHT)]
                + [*zip(xs_ground, ys_ground)]
            )
            ground.set_color(237 / 255, 201 / 255, 175 / 255)
            oil = rendering.make_polyline(
                [
                    (self.friction_start, ROAD_HIGHT - 0.1),
                    (self.goal_zone[1], ROAD_HIGHT - 0.1),
                ]
            )
            oil.set_linewidth(2)
            xs_sea = np.linspace(self.goal_zone[1], self.max_position, 25)
            ys_sea = np.maximum(np.sin(xs_sea * 7.1) * 2.0, 0.2)
            sea = rendering.make_polygon(
                [*zip(xs_sea, ys_sea)] + [(xs_sea[-1], 0.0), (xs_sea[0], 0.0)]
            )
            sea.set_color(0, 105 / 255, 148 / 255)
            sun = rendering.make_circle(2.5)
            sun.set_color(252 / 255, 212 / 255, 64 / 255)
            sun.add_attr(rendering.Transform((65, 35)))
            car_width, car_height = CAR_WIDTH, CAR_HEIGHT
            l, r, t, b = -car_width / 2, car_width / 2, car_height, 0
            car = rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])
            self.car_transform = rendering.Transform()
            car.add_attr(self.car_transform)
            radius = car_height / 2.5
            frontwheel = rendering.make_circle(radius)
            frontwheel.add_attr(rendering.Transform(translation=(car_width / 4, 0)))
            frontwheel.add_attr(self.car_transform)
            frontwheel.set_color(0.5, 0.5, 0.5)
            frontwheel_rim = rendering.make_circle(0.3, res=30, filled=True)
            frontwheel_rim.set_color(1.0, 0.0, 0.0)
            frontwheel_rim.add_attr(rendering.Transform(translation=(radius - 0.3, 0)))
            self.frontwheel_rim_transform = rendering.Transform()
            frontwheel_rim.add_attr(self.frontwheel_rim_transform)
            backwheel = rendering.make_circle(radius)
            backwheel.add_attr(rendering.Transform(translation=(-car_width / 4, 0)))
            backwheel.add_attr(self.car_transform)
            backwheel.set_color(0.5, 0.5, 0.5)
            backwheel_rim = rendering.make_circle(0.3, res=30, filled=True)
            backwheel_rim.set_color(1.0, 0.0, 0.0)
            backwheel_rim.add_attr(rendering.Transform(translation=(radius - 0.3, 0)))
            self.backwheel_rim_transform = rendering.Transform()
            backwheel_rim.add_attr(self.backwheel_rim_transform)

            def make_flag(position):
                flagx = position
                flagy1 = ROAD_HIGHT
                flagy2 = flagy1 + 2.0
                flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
                flag = rendering.FilledPolygon(
                    [
                        (flagx, flagy2),
                        (flagx, flagy2 - 1.0),
                        (flagx + 2.5, flagy2 - 0.5),
                    ]
                )
                flag.set_color(0.8, 0.8, 0)
                return flag, flagpole

            right_flag, right_flagpole = make_flag(self.goal_zone[0])
            left_flag, left_flagpole = make_flag(self.goal_zone[1])
            self.viewer.add_geom(sky)
            self.viewer.add_geom(sea)
            self.viewer.add_geom(ground)
            self.viewer.add_geom(oil)
            self.viewer.add_geom(sun)
            self.viewer.add_geom(car)
            self.viewer.add_geom(frontwheel)
            self.viewer.add_geom(frontwheel_rim)
            self.viewer.add_geom(backwheel)
            self.viewer.add_geom(backwheel_rim)
            self.viewer.add_geom(right_flagpole)
            self.viewer.add_geom(right_flag)
            self.viewer.add_geom(left_flagpole)
            self.viewer.add_geom(left_flag)
        pos = self.car.hull.position
        self.car_transform.set_translation(*pos)
        self.car_transform.set_rotation(self.car.hull.angle)
        self.frontwheel_rim_transform.set_translation(*pos)
        self.backwheel_rim_transform.set_translation(*pos)
        self.frontwheel_rim_transform.set_translation(*self.car.wheels[0].position)
        self.frontwheel_rim_transform.set_rotation(self.car.wheels[0].angle)
        self.backwheel_rim_transform.set_rotation(self.car.wheels[1].angle)
        self.backwheel_rim_transform.set_translation(*self.car.wheels[1].position)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    from pyglet.window import key
    from gym.wrappers import TimeLimit

    a = np.array([0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.SPACE:
            a[1] = +0.8

    def key_release(k, mod):
        if k == key.RIGHT:
            a[0] = 0
        if k == key.LEFT:
            a[0] = 0
        if k == key.SPACE:
            a[1] = 0

    env: gym.Env = CliffDaredevil()
    env = TimeLimit(env, 600)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, truncated, info = env.step(a)
            total_reward += r
            total_cost += info["cost"]
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.4f}".format(x) for x in a]))
                print("step {} total_reward {:+0.4f}".format(steps, total_reward))
                print("step {} total_cost {:+0.4f}".format(steps, total_cost))
                print("step {} cost {:+0.4f}".format(steps, info["cost"]))
                print("step {} reward {:+0.4f}".format(steps, r))
            steps += 1
            isopen, _ = env.render()
            if done or restart or not isopen:
                break
    env.close()
