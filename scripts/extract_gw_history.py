# -*- coding: utf-8 -*-
# extract_gw_history.py
# Abaqus Python script to extract guided wave sensor history from ODB.
#
# Extracts U3 (out-of-plane displacement) time histories at sensor nodes
# and writes CSV files for post-processing.
#
# Strategy: Use ODB node coordinates to correctly identify sensor positions,
# even when partitioning changes node numbering or region ordering.
#
# Usage: abaqus python extract_gw_history.py Job-GW-Healthy.odb Job-GW-Debond.odb

import sys
import os
import csv
import math
from odbAccess import openOdb


def _get_sensor_node_map(odb):
    """Build sensor_id -> (node_label, x, y, z) from assembly node sets.

    Returns dict {sensor_id: {'label': int, 'x': float, 'y': float, 'z': float}}
    """
    sensor_map = {}
    root_assembly = odb.rootAssembly

    # Look for Set-Sensor-N in assembly-level node sets
    for set_name in root_assembly.nodeSets.keys():
        for i in range(10):
            target = 'SET-SENSOR-%d' % i
            if set_name.upper() == target:
                ns = root_assembly.nodeSets[set_name]
                # nodeSets contain nodes per instance
                for inst_name_nodes in ns.nodes:
                    for node in inst_name_nodes:
                        coords = node.coordinates
                        sensor_map[i] = {
                            'label': node.label,
                            'instance': '',  # filled below
                            'x': coords[0],
                            'y': coords[1],
                            'z': coords[2]
                        }

    # If assembly-level sets not found, try instance-level
    if not sensor_map:
        for inst_name, inst in root_assembly.instances.items():
            for set_name in inst.nodeSets.keys():
                for i in range(10):
                    target = 'SET-SENSOR-%d' % i
                    if set_name.upper() == target:
                        ns = inst.nodeSets[set_name]
                        for node in ns.nodes:
                            coords = node.coordinates
                            sensor_map[i] = {
                                'label': node.label,
                                'instance': inst_name,
                                'x': coords[0],
                                'y': coords[1],
                                'z': coords[2]
                            }

    return sensor_map


def extract_sensor_history(odb_path, output_dir=None):
    """Extract U3 history output from all sensor sets in ODB.

    Uses node coordinates to correctly identify sensor positions.
    Returns dict: {sensor_id: [(time, u3), ...]}
    Also writes CSV: <odb_name>_sensors.csv
    """
    if not os.path.exists(odb_path):
        print("ERROR: ODB not found: %s" % odb_path)
        return None

    odb_name = os.path.splitext(os.path.basename(odb_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(odb_path) or '.'

    print("Opening ODB: %s" % odb_path)
    odb = openOdb(odb_path, readOnly=True)

    # Step 1: Get sensor node map from assembly sets
    sensor_map = _get_sensor_node_map(odb)
    if sensor_map:
        print("  Found %d sensor sets in assembly:" % len(sensor_map))
        for sid in sorted(sensor_map.keys()):
            info = sensor_map[sid]
            print("    Sensor %d: node=%d, pos=(%.1f, %.1f, %.1f)" % (
                sid, info['label'], info['x'], info['y'], info['z']))
    else:
        print("  WARNING: No sensor node sets found in assembly")

    # Detect curved geometry: check if sensors are on a cylinder
    # (if all sensors have similar sqrt(x^2 + z^2), it's curved)
    is_curved = False
    if len(sensor_map) >= 2:
        radii = []
        for sid, info in sensor_map.items():
            r = math.sqrt(info['x'] ** 2 + info['z'] ** 2)
            radii.append(r)
        if radii and min(radii) > 100:  # sensors far from origin = cylindrical
            r_mean = sum(radii) / len(radii)
            r_spread = max(radii) - min(radii)
            if r_spread / r_mean < 0.01:  # within 1% = cylindrical
                is_curved = True
                print("  Detected CURVED geometry (R_mean=%.0f mm)" % r_mean)

    # Step 2: Build node_label -> sensor_id lookup
    # History region keys look like: "Node PART-OUTERSKIN-1.12345" or "Node ASSEMBLY.12345"
    label_to_sensor = {}
    for sid, info in sensor_map.items():
        label_to_sensor[info['label']] = sid

    # Step 3: Extract history data
    step = odb.steps['Step-Wave']
    history_regions = step.historyRegions

    sensor_data = {}
    sensor_components = {}  # sensor_id -> {'U1': data, 'U2': data, 'U3': data}
    region_assignments = []

    for region_name, region in history_regions.items():
        # Parse node label from region name
        # Formats: "Node PART-OUTERSKIN-1.12345", "Node ASSEMBLY.SET-SENSOR-0"
        node_label = None
        sensor_id = None

        # Try direct sensor set name match first
        for i in range(10):
            set_name = 'Set-Sensor-%d' % i
            if set_name.upper() in region_name.upper():
                sensor_id = i
                break

        # Try parsing node label from region name
        if sensor_id is None and '.' in region_name:
            try:
                parts = region_name.split('.')
                label_str = parts[-1].strip()
                node_label = int(label_str)
                if node_label in label_to_sensor:
                    sensor_id = label_to_sensor[node_label]
            except (ValueError, IndexError):
                pass

        # If we found a matching sensor, extract displacement components
        if sensor_id is not None:
            if sensor_id not in sensor_components:
                sensor_components[sensor_id] = {}

            for var_name in region.historyOutputs.keys():
                for comp in ['U1', 'U2', 'U3']:
                    if comp in var_name:
                        ho = region.historyOutputs[var_name]
                        data = list(ho.data)
                        sensor_components[sensor_id][comp] = data
                        if comp == 'U3':
                            # For flat panel compatibility, U3 is still the primary
                            if sensor_id not in sensor_data:
                                sensor_data[sensor_id] = data
                                region_assignments.append(
                                    (sensor_id, region_name, len(data), var_name))

    # For curved geometry: compute radial displacement from U1, U3 components
    if is_curved and sensor_components:
        print("  Computing radial displacement (U_r) for curved geometry...")
        for sid in sorted(sensor_components.keys()):
            if sid not in sensor_map:
                continue
            info = sensor_map[sid]
            x, z = info['x'], info['z']
            r = math.sqrt(x * x + z * z)
            if r < 1.0:
                continue
            cos_t = x / r
            sin_t = z / r

            comps = sensor_components[sid]
            u1_data = comps.get('U1', [])
            u3_data = comps.get('U3', [])

            if u1_data and u3_data:
                n_pts = min(len(u1_data), len(u3_data))
                ur_data = []
                for j in range(n_pts):
                    t_val = u1_data[j][0]
                    u1_val = u1_data[j][1]
                    u3_val = u3_data[j][1]
                    # Radial displacement: projection onto radial direction
                    u_r = u1_val * cos_t + u3_val * sin_t
                    ur_data.append((t_val, u_r))
                sensor_data[sid] = ur_data
                print("    Sensor %d: U_r = U1*cos(%.2f) + U3*sin(%.2f)" % (
                    sid, math.degrees(math.atan2(sin_t, cos_t)),
                    math.degrees(math.atan2(sin_t, cos_t))))

    # Fallback: if no sensors matched, try coordinate-based matching
    if not sensor_data:
        print("  Primary matching failed, trying coordinate-based fallback...")

        # Collect all history regions with U3 data and their node labels
        unmatched_regions = []
        for region_name, region in history_regions.items():
            for var_name in region.historyOutputs.keys():
                if 'U3' in var_name:
                    ho = region.historyOutputs[var_name]
                    data = list(ho.data)
                    # Try to get node label
                    node_label = None
                    if '.' in region_name:
                        try:
                            label_str = region_name.split('.')[-1].strip()
                            node_label = int(label_str)
                        except (ValueError, IndexError):
                            pass
                    unmatched_regions.append({
                        'region_name': region_name,
                        'var_name': var_name,
                        'data': data,
                        'node_label': node_label
                    })

        # Get node coordinates from ODB mesh
        node_coords = {}
        root_assembly = odb.rootAssembly
        for inst_name, inst in root_assembly.instances.items():
            if 'OUTERSKIN' in inst_name.upper() or 'OUTER' in inst_name.upper():
                for node in inst.nodes:
                    node_coords[node.label] = node.coordinates

        # Match by finding which region's node is closest to each sensor position
        sensor_offsets = [0.0, 50.0, 100.0, 150.0, 200.0]
        for ur in unmatched_regions:
            if ur['node_label'] is not None and ur['node_label'] in node_coords:
                coords = node_coords[ur['node_label']]
                ur['x'] = coords[0]
                ur['y'] = coords[1]
                ur['z'] = coords[2]
            else:
                ur['x'] = None

        # Sort unmatched regions by X coordinate and assign sensor IDs
        regions_with_coords = [ur for ur in unmatched_regions if ur['x'] is not None]
        regions_with_coords.sort(key=lambda r: r['x'])

        for idx, ur in enumerate(regions_with_coords):
            sensor_data[idx] = ur['data']
            region_assignments.append(
                (idx, ur['region_name'], len(ur['data']), ur['var_name']))
            print("  Fallback sensor %d: node=%s, x=%.1f, region=%s" % (
                idx, ur['node_label'], ur['x'], ur['region_name'][:60]))

    odb.close()

    # Print assignment summary
    print("\n  Sensor assignments:")
    for sid, rname, npts, vname in sorted(region_assignments):
        x_pos = sensor_map[sid]['x'] if sid in sensor_map else '?'
        print("    Sensor %d: %d points, x=%s, region=%s" % (
            sid, npts, x_pos, rname[:60]))

    if not sensor_data:
        print("WARNING: No U3 history data found in %s" % odb_path)
        return None

    # Step 4: Write CSV, sorted by sensor ID
    csv_path = os.path.join(output_dir, odb_name + '_sensors.csv')
    max_len = max(len(v) for v in sensor_data.values())
    sensor_ids = sorted(sensor_data.keys())

    # Write sensor positions as comment in first line
    # For curved: use arc distance from sensor 0; for flat: use X position
    actual_x = []
    if is_curved and 0 in sensor_map:
        # Compute arc distance from sensor 0
        s0 = sensor_map[0]
        r0 = math.sqrt(s0['x'] ** 2 + s0['z'] ** 2)
        theta0 = math.atan2(s0['z'], s0['x'])
        for sid in sensor_ids:
            if sid in sensor_map:
                si = sensor_map[sid]
                ri = math.sqrt(si['x'] ** 2 + si['z'] ** 2)
                theta_i = math.atan2(si['z'], si['x'])
                arc_dist = r0 * (theta_i - theta0)
                actual_x.append('%.1f' % arc_dist)
            else:
                actual_x.append('?')
        print("  Using arc distance from sensor 0 for positions")
    else:
        for sid in sensor_ids:
            if sid in sensor_map:
                actual_x.append('%.1f' % sensor_map[sid]['x'])
            else:
                actual_x.append('?')

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        # Header with actual positions as comment
        disp_label = 'Ur' if is_curved else 'U3'
        header = ['time_s'] + ['sensor_%d_%s' % (sid, disp_label)
                                for sid in sensor_ids]
        writer.writerow(header)
        # Second row: actual X positions (for reference)
        pos_row = ['# x_mm'] + actual_x
        writer.writerow(pos_row)

        for ti in range(max_len):
            row = []
            first_sensor = sensor_ids[0]
            if ti < len(sensor_data[first_sensor]):
                t = sensor_data[first_sensor][ti][0]
            else:
                t = 0.0
            row.append('%.9e' % t)
            for sid in sensor_ids:
                if ti < len(sensor_data[sid]):
                    row.append('%.9e' % sensor_data[sid][ti][1])
                else:
                    row.append('')
            writer.writerow(row)

    print("\nCSV written: %s (%d rows x %d sensors)" % (
        csv_path, max_len, len(sensor_ids)))

    # Return sensor_data with metadata
    return {'data': sensor_data, 'sensor_map': sensor_map,
            'is_curved': is_curved}


def compute_group_velocity(result, sensor_offsets=None):
    """Estimate group velocity from peak arrival times.

    Uses actual sensor coordinates from ODB if available.
    """
    if result is None:
        return []

    sensor_data = result['data']
    sensor_map = result.get('sensor_map', {})
    is_curved = result.get('is_curved', False)

    # Compute sensor distances (X for flat, arc for curved)
    if is_curved and sensor_map and 0 in sensor_map:
        s0 = sensor_map[0]
        r0 = math.sqrt(s0['x'] ** 2 + s0['z'] ** 2)
        theta0 = math.atan2(s0['z'], s0['x'])
        actual_offsets = {}
        for sid in sorted(sensor_data.keys()):
            if sid in sensor_map:
                si = sensor_map[sid]
                theta_i = math.atan2(si['z'], si['x'])
                actual_offsets[sid] = r0 * (theta_i - theta0)
            else:
                actual_offsets[sid] = sid * 30.0
        print("  Using arc distance for velocity calculation")
    elif sensor_map:
        actual_offsets = {}
        for sid in sorted(sensor_data.keys()):
            if sid in sensor_map:
                actual_offsets[sid] = sensor_map[sid]['x']
            elif sensor_offsets and sid < len(sensor_offsets):
                actual_offsets[sid] = sensor_offsets[sid]
            else:
                actual_offsets[sid] = sid * 50.0
    elif sensor_offsets:
        actual_offsets = {i: sensor_offsets[i]
                         for i in range(len(sensor_offsets))
                         if i in sensor_data}
    else:
        actual_offsets = {sid: sid * 50.0 for sid in sensor_data.keys()}

    print("\n--- Group Velocity Estimation ---")
    peak_times = {}
    for sid in sorted(sensor_data.keys()):
        data = sensor_data[sid]
        max_abs = 0.0
        t_peak = 0.0
        for t, u3 in data:
            if abs(u3) > max_abs:
                max_abs = abs(u3)
                t_peak = t
        peak_times[sid] = t_peak
        x_pos = actual_offsets.get(sid, 0.0)
        disp_label = '|Ur|' if is_curved else '|U3|'
        pos_label = 'arc' if is_curved else 'x'
        print("  Sensor %d (%s=%.1f mm): peak at t=%.3e s, %s_max=%.3e" % (
            sid, pos_label, x_pos, t_peak, disp_label, max_abs))

    # Velocity from consecutive sensor pairs (skip sensor at excitation)
    print("\n  Group velocity (from sensor pairs):")
    sids = sorted(sensor_data.keys())
    velocities = []
    for j in range(len(sids) - 1):
        sid1, sid2 = sids[j], sids[j + 1]
        x1 = actual_offsets.get(sid1, 0.0)
        x2 = actual_offsets.get(sid2, 0.0)
        dt = peak_times[sid2] - peak_times[sid1]
        dx = x2 - x1  # mm
        if abs(dx) < 1.0:
            print("    Sensor %d->%d: dx~0 (same position), skip" % (sid1, sid2))
            continue
        if abs(dt) > 1e-10:
            v = dx / dt / 1000.0  # m/s
            velocities.append(v)
            print("    Sensor %d->%d: dx=%.0f mm, dt=%.3e s -> v=%.0f m/s" % (
                sid1, sid2, dx, dt, v))
        else:
            print("    Sensor %d->%d: dt~0 (near-field)" % (sid1, sid2))

    if velocities:
        v_avg = sum(velocities) / len(velocities)
        print("\n  Average group velocity: %.0f m/s" % v_avg)
        print("  Theory (A0 at 50 kHz):  ~1550 m/s")
        print("  Deviation: %.1f%%" % (abs(v_avg - 1550) / 1550 * 100))
    return velocities


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: abaqus python extract_gw_history.py <odb1> [odb2] ...")
        sys.exit(1)

    all_results = {}
    for odb_path in sys.argv[1:]:
        print("\n" + "=" * 60)
        result = extract_sensor_history(odb_path)
        if result:
            odb_name = os.path.splitext(os.path.basename(odb_path))[0]
            all_results[odb_name] = result
            compute_group_velocity(result)

    print("\n" + "=" * 60)
    print("Extraction complete: %d ODB(s) processed" % len(all_results))
