import cozmo
import asyncio

from Common.colors import Colors

from PIL import Image


# region CozmoPhotoStream Class
class CozmoPhotoStream:
    def __init__(self):
        self.robot: cozmo.robot.Robot = None
        self.cubes = []
        camera_image = Image.open('res/FaceImages/camera.png').resize(cozmo.oled_face.dimensions(), Image.BICUBIC)
        self.face_image = cozmo.oled_face.convert_image_to_screen_data(camera_image, invert_image=True)
        self.latest_image: Image = None

    async def on_camera_image(self, evt=None, **kwargs):
        # Grab the latest frame and save the raw image as a greyscale png file
        if self.robot.world.latest_image is not None:
            self.latest_image: Image = self.robot.world.latest_image.raw_image.convert('RGB')

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

        self.robot.set_all_backpack_lights(Colors.GREEN)

        # Add event handler to process each image as it becomes available
        self.robot.camera.add_event_handler(cozmo.robot.camera.EvtNewRawCameraImage, self.on_camera_image)

        # self.robot.display_oled_face_image(self.face_image, 2000, in_parallel=True)

        # await self.robot_say('Awaiting a command')

        while True:
            # Wait for SIG-INT and keep the thread controlling Cozmo alive
            await asyncio.sleep(0.5)


# endregion

photo_stream = CozmoPhotoStream()


async def cozmo_program(robot: cozmo.robot.Robot):
    photo_stream.robot = robot
    await photo_stream.run()


def run_cozmo_photostream():
    cozmo.run_program(cozmo_program, use_viewer=False)
