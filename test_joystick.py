import pygame
import time,os
os.environ["SDL_JOYSTICK_DEVICE"] = "/dev/input/js0"

pygame.init()
pygame.joystick.init()

# 打印pygame识别到的游戏杆数量
print(f"pygame识别到的游戏杆数量: {pygame.joystick.get_count()}")

if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"✅ 成功访问: {joystick.get_name()}")
    print(f"轴数量: {joystick.get_numaxes()}, 按钮数量: {joystick.get_numbuttons()}")
    while True:
        pygame.event.pump()
        left_x = joystick.get_axis(0)
        left_y = joystick.get_axis(1)
        print(f"左摇杆(X,Y): ({left_x:.2f}, {left_y:.2f})", end='\r')
        time.sleep(0.01)
else:
    print("❌ pygame未识别到任何游戏杆")