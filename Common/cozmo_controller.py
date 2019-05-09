import cozmo
import asyncio

from Common.colors import Colors
from queue import Queue
from threading import Lock
from cozmo.util import degrees, distance_mm, speed_mmps

from PIL import Image


# region CozmoController Class
class CozmoController:
    def __init__(self):
        self.robot: cozmo.robot.Robot = None
        self.cubes = []
        camera_image = Image.open('res/FaceImages/camera.png').resize(cozmo.oled_face.dimensions(), Image.BICUBIC)
        fist_image = Image.open('res/FaceImages/fist.png').resize(cozmo.oled_face.dimensions(), Image.BICUBIC)
        ok_image = Image.open('res/FaceImages/ok.png').resize(cozmo.oled_face.dimensions(), Image.BICUBIC)
        self.camera_image = cozmo.oled_face.convert_image_to_screen_data(camera_image, invert_image=True)
        self.fist_image = cozmo.oled_face.convert_image_to_screen_data(fist_image, invert_image=True)
        self.ok_image = cozmo.oled_face.convert_image_to_screen_data(ok_image, invert_image=True)
        self.latest_image: Image = None
        self.command_run_lock: Lock = Lock()
        self.command_q: Queue = Queue(maxsize=1)

    async def on_camera_image(self, evt=None, **kwargs):
        # Grab the latest frame and save the raw image as a greyscale png file
        if self.robot.world.latest_image is not None:
            self.latest_image: Image = self.robot.world.latest_image.raw_image.convert('RGB')

    async def robot_say(self, text):
        await self.robot.say_text(text, duration_scalar=0.6).wait_for_completed()

    async def add_photo_event_handler(self):
        # Add event handler to process each image as it becomes available
        self.robot.camera.add_event_handler(cozmo.robot.camera.EvtNewRawCameraImage, self.on_camera_image)

    async def run_command(self, command: str):
        print('Command received:', command)
        if command == 'claw':
            await self.robot.play_anim(name='anim_codelab_frightenedcozmo_01').wait_for_completed()
        elif command == 'fist':
            await self.robot_say('Fist bump!')
            await self.robot.display_oled_face_image(self.fist_image, 2000, in_parallel=True).wait_for_completed()
        elif command == 'ok':
            await self.robot_say('Ok!')
            await self.robot.display_oled_face_image(self.ok_image, 2000, in_parallel=True).wait_for_completed()
        elif command == 'peace_sign':
            await self.robot.turn_in_place(degrees(360)).wait_for_completed()
        elif command == 'open_hand':
            await self.robot.play_anim(name='anim_pounce_success_01').wait_for_completed()
            await self.robot.drive_straight(distance_mm(65), speed_mmps(150)).wait_for_completed()
        elif command == 'letter_y':
            # Stack two cubes
            # Look around until Cozmo knows where at least 2 cubes are
            lookaround = self.robot.start_behavior(cozmo.behavior.BehaviorTypes.LookAroundInPlace)
            self.cubes = await self.robot.world.wait_until_observe_num_objects(num=2,
                                                                               object_type=cozmo.objects.LightCube,
                                                                               timeout=60)
            lookaround.stop()
            if len(self.cubes) < 2:
                await self.robot_say(
                    'I need to be able to see at least two cubes to do that, please place them where I can find them ')
            else:
                # Try and pickup the 1st cube
                current_action = self.robot.pickup_object(self.cubes[0], num_retries=3)
                await current_action.wait_for_completed()
                if current_action.has_failed:
                    code, reason = current_action.failure_reason
                    result = current_action.result
                    await self.robot.set_lift_height(0.0, in_parallel=False).wait_for_completed()
                    await self.robot.set_head_angle(degrees(25), in_parallel=False).wait_for_completed()
                    print(f'Pickup Cube failed: code={code} reason={reason} result={result}')
                    return

                # Now try to place that cube on the 2nd one
                current_action = self.robot.place_on_object(self.cubes[1], num_retries=3)
                await current_action.wait_for_completed()
                await self.robot.set_lift_height(0.0, in_parallel=False).wait_for_completed()
                await self.robot.set_head_angle(degrees(25), in_parallel=False).wait_for_completed()
                if current_action.has_failed:
                    code, reason = current_action.failure_reason
                    result = current_action.result
                    print(f'Place On Cube failed: code={code} reason={reason} result={result}')
                    return
        else:
            await self.robot_say(command.replace('_', ' '))
        # Move the head back up after any animations to allow for better hand framing
        await self.robot.set_head_angle(degrees(25), in_parallel=False).wait_for_completed()

    async def run(self):
        await self.robot.set_head_angle(degrees(25), in_parallel=False).wait_for_completed()
        # Turn backpack lights to RED
        self.robot.set_all_backpack_lights(Colors.RED)

        await self.robot.world.connect_to_cubes()

        # Move lift out of the way of the camera
        await self.robot.set_lift_height(0.0).wait_for_completed()

        # Settings for signals from Cozmo's camera
        self.robot.camera.image_stream_enabled = True
        self.robot.camera.color_image_enabled = True

        self.robot.set_all_backpack_lights(Colors.GREEN)

        # Add event handler to process each image as it becomes available
        # self.robot.camera.add_event_handler(cozmo.robot.camera.EvtNewRawCameraImage, self.on_camera_image)

        self.robot.display_oled_face_image(self.camera_image, 2000, in_parallel=True)

        # await self.robot_say('Awaiting commands')

        while True:
            # Wait for SIG-INT and keep the thread controlling Cozmo alive
            if not self.command_q.empty():
                external_command = self.command_q.get()
                # Acquire lock to prevent further commands being pushed until the current command finishes
                self.command_run_lock.acquire()
                await self.run_command(external_command)
                # Release the lock so further commands can be run
                self.command_run_lock.release()
            else:
                await asyncio.sleep(0.5)


# endregion

cozmo_controller = CozmoController()


async def cozmo_program(robot: cozmo.robot.Robot):
    cozmo_controller.robot = robot

    # Run photo streaming and main command collection loop concurrently
    await asyncio.gather(
        cozmo_controller.run(),
        cozmo_controller.add_photo_event_handler()
    )


def run_cozmo_controller():
    cozmo.run_program(cozmo_program, use_viewer=False, )
