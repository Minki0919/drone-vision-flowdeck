"""Read-only Flow Deck simulation panel; never feeds data to the controller."""
import cv2
import numpy as np


def sample_state(stamp, now, received_wall, wall_now, clock_wall, limit=.15):
    if stamp is None:
        return 'WAITING'
    if now is None:
        return 'WAITING_CLOCK'
    if now < stamp:
        return 'FUTURE'
    if (received_wall is None or wall_now-received_wall >= .25
            or clock_wall is None or wall_now-clock_wall >= .5 or now-stamp > limit):
        return 'STALE'
    return 'LIVE'


def render_flow_panel(flow, gray, image_stamp, state, now, wall_now, tof, pilot):
    """Render captured pixels and measured telemetry, including stale/invalid states."""
    canvas = np.zeros((660, 800, 3), np.uint8)
    green, amber = (140, 240, 190), (0, 190, 255)

    def label(text, x, y, color=green, scale=.46):
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1., (790-x)/max(1, width))
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    image_status = sample_state(image_stamp, now, state['flow_image_wall'], wall_now, state['clock_wall'])
    if gray is None and image_status == 'LIVE':
        image_status = 'INVALID'
    attitude = flow.attitudes[-1] if flow.attitudes else None
    imu_status = sample_state(None if attitude is None else attitude[0], now,
                             state['flow_imu_wall'], wall_now, state['clock_wall'])
    rotation = None if attitude is None else attitude[1]
    if rotation is None and imu_status == 'LIVE':
        imu_status = 'INVALID'
    velocity_status = sample_state(flow.velocity_stamp, now, state['flow_image_wall'],
                                   wall_now, state['clock_wall'])
    velocity_valid = (velocity_status == 'LIVE' and image_status == 'LIVE'
                      and imu_status == 'LIVE' and tof['valid'] and flow.velocity is not None)
    if velocity_status == 'LIVE' and not velocity_valid:
        velocity_status = 'UNAVAILABLE'
    label('FLOW DECK | GAZEBO SENSOR PROXY | NOT HARDWARE', 12, 26)
    label('Downward camera 128 x 128 | '+image_status, 12, 55,
          green if image_status == 'LIVE' else amber)
    if gray is not None:
        canvas[70:390, 12:332] = cv2.cvtColor(
            cv2.resize(gray, (320, 320), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    else:
        label('No decodable camera frame', 20, 225, amber)
    label('DOWNWARD ToF | '+tof['state'], 352, 90,
          green if tof['valid'] else amber)
    distance, age = tof['distance_m'], tof['age_s']
    label('-- m' if distance is None else f'{distance:.3f} m / {distance*1000:.0f} mm', 352, 122)
    label('Age: --' if age is None else f'Age: {age:.3f} s', 352, 150)
    label('Control range: 0.02 <= d < 2.00 m', 352, 178)
    cv2.rectangle(canvas, (365, 200), (405, 375), (90, 90, 90), 1)
    if tof['valid']:
        top = 375-round(174*np.clip(distance/2., 0., 1.))
        cv2.rectangle(canvas, (366, top), (404, 374), green, -1)
    label('2.00 m', 420, 212)
    label('Surface distance', 420, 285)
    label('0.00 m', 420, 375)
    label('Body velocity (forward / left / up): '+velocity_status, 12, 420,
          green if velocity_valid else amber)
    label('-- m/s' if not velocity_valid else
          ' / '.join(f'{value:+.3f}' for value in flow.velocity)+' m/s', 12, 446)
    label('Flow: '+flow.reason, 12, 472)
    angles = '-- / -- / --'
    if imu_status == 'LIVE':
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(np.clip(-rotation[2, 0], -1., 1.))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        angles = ' / '.join(f'{value:+.1f}' for value in np.rad2deg([roll, pitch, yaw]))
    label(f'IMU: {imu_status} | roll / pitch / yaw (deg): {angles}', 12, 498)
    status = flow.telemetry()
    label('Last controller assist | XY damping: '+str(status.get('horizontal_assist', False))+
          ' | height hold: '+str(status.get('height_hold', False))+
          ' | soft descent: '+str(status.get('touchdown_softening', False)), 12, 524)
    label('Ring ToF: '+pilot.ring_tof.get('state', 'WAIT')+
          ' | Truck ToF: '+pilot.truck_tof.get('state', 'WAIT'), 12, 550)
    label('Phase: '+pilot.phase+' | capture: '+str(image_stamp)+' s', 12, 576)
    label('STALE = last frame only; unavailable velocity / attitude shown as --', 12, 610, amber)
    label('Q / Esc or close a sensor window: stop simulation', 12, 640)
    return canvas
