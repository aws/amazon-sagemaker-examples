def vrp_action_go_from_a_to_b(a, b):
    # 0: Up, 1: Down, 2: Left, 3: Right
    action = 0
    cur_x = a[0]
    cur_y = a[1]
    tar_x = b[0]
    tar_y = b[1]

    x_diff = tar_x - cur_x
    y_diff = tar_y - cur_y

    if abs(x_diff) >= abs(y_diff):
        # Move horizontally
        if x_diff > 0:
            action = 4
        elif x_diff < 0:
            action = 3
    else:
        # Move vertically
        if y_diff > 0:
            action = 1
        elif y_diff < 0:
            action = 2

    return action
