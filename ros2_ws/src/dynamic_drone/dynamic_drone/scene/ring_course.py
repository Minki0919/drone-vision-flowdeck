"""Stage-2 air gates and building barriers; rendering/collision geometry only."""
import xml.etree.ElementTree as ET

from dynamic_drone.scene.environment import solid, text


def add_ring_corridor(world, gate):
    """Keep existing rings and close bypass lanes outside their openings."""
    # The circular rim remains visible inside a square air-gate frame.
    # Its corner gaps are narrower than the aircraft; the opening stays empty.
    half_opening = gate['width']/2 + gate['frame_thickness']/2
    frame = .18
    outer = half_opening + frame
    height = 4.5
    buildings = []
    for index, (x, y, z) in enumerate(gate['centers'], 1):
        prefix = f'stage2_gate_{index}'
        for side in (-1, 1):
            solid(world, f'{prefix}_post_{side}',
                  [x, y+side*(half_opening+frame/2), z],
                  [.24, frame, 2*half_opening], color=(.65, .45, .15))
            edge = y+side*outer
            boundary = side*4.
            name = f'{prefix}_building_{side}'
            depth = 1.0 if (index+side) % 2 else 1.4
            model = solid(world, name, [x, (edge+boundary)/2, height/2],
                          [depth, abs(boundary-edge), height], color=(.48, .50, .53))
            buildings.append(name)
            # Facade windows give the front camera stationary visual features.
            link = model.find('link')
            for row in range(5):
                for col in range(4):
                    window = ET.SubElement(link, 'visual', name=f'window_{row}_{col}')
                    width = abs(boundary-edge)
                    text(window, 'pose', f'{-depth/2-.002} {(col-1.5)*width/4} {(row-2)*.7} 0 0 0')
                    box = ET.SubElement(ET.SubElement(window, 'geometry'), 'box')
                    text(box, 'size', f'.002 {min(.30, width/6)} .34')
                    material = ET.SubElement(window, 'material')
                    for key in ('ambient', 'diffuse'):
                        text(material, key, '.10 .14 .18 1')
        for label, bottom, top in [('base', 0., z-half_opening),
                                   ('lintel', z+half_opening, height)]:
            solid(world, f'{prefix}_{label}', [x, y, (bottom+top)/2],
                  [.24, 2*outer, top-bottom], color=(.65, .45, .15))
    return dict(name='ring_air_gates_with_building_barriers', stages_modified=[2],
                ring_count=len(gate['centers']), building_count=len(buildings),
                physical_collision_geometry=True, ring_openings_preserved=True,
                positions_use='rendering_and_diagnostics_only_not_controller_input')
