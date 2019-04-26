import cozmo
import asyncio

from Common.colors import Colors

from cozmo.util import distance_mm
from cozmo.objects import LightCube
from PIL import Image


# region CubePhoto Class
class CubePhoto:
    def __init__(self, robot: cozmo.robot.Robot):
        self.robot = robot
        self.cubes = []
        self.image_num = 0
        camera_image = Image.open('../res/FaceImages/camera.png').resize(cozmo.oled_face.dimensions(), Image.BICUBIC)
        self.face_image = cozmo.oled_face.convert_image_to_screen_data(camera_image, invert_image=True)

    async def on_cube_tapped(self, evt, obj: LightCube, **kwargs):
        obj.set_lights(Colors.RED)

        # Display camera icon
        self.robot.display_oled_face_image(self.face_image, 1000.0, in_parallel=True)

        # Grab the latest frame and save the raw image as a greyscale png file
        img_latest: Image = self.robot.world.latest_image.raw_image
        img_latest.save(f'../res/TrainingData/{self.image_num}.png', 'png')
        self.image_num += 1

        obj.set_lights(Colors.BLUE)

    async def robot_say(self, text):
        await self.robot.say_text(text, duration_scalar=0.6).wait_for_completed()

    async def run(self):
        # Turn backpack lights to RED
        self.robot.set_all_backpack_lights(Colors.RED)

        # Move lift out of the way of the camera
        await self.robot.set_lift_height(0.0).wait_for_completed()

        # Settings for signals from Cozmo's camera
        self.robot.camera.image_stream_enabled = True
        self.robot.camera.color_image_enabled = True

        # Connect to the cubes
        await self.robot.world.connect_to_cubes()

        # Begin looking around for objects
        await self.robot_say('Scanning for cubes')
        look_around = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)

        # Find the cubes and store their information
        self.cubes = await self.robot.world.wait_until_observe_num_objects(num=3, object_type=LightCube, timeout=120)

        look_around.stop()

        await self.robot_say('I found the cubes')
        print(f'Found cubes: {self.cubes}')

        self.robot.set_all_backpack_lights(Colors.GREEN)

        for cube in self.cubes:
            # Set the lights on all the found cubes
            cube.set_lights(Colors.BLUE)

            # Add cube tap event handler with callback
            cube.add_event_handler(cozmo.objects.EvtObjectTapped, self.on_cube_tapped)

        await self.robot_say('Tap any cube to take a photo for the dataset')


# endregion


async def cozmo_program(robot: cozmo.robot.Robot):
    cube_photo = CubePhoto(robot)
    await cube_photo.run()

    # Wait to receive keyboard interrupt command to exit (CTRL-C)
    while True:
        await asyncio.sleep(0.5)


cozmo.run_program(cozmo_program, use_viewer=True, force_viewer_on_top=True)
