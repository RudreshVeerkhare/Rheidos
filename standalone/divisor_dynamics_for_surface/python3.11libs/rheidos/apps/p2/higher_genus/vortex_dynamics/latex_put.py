H = 145
w_offset_lower = 0
h_offset_lower = 10

print(f"""
    
    \footnotesize

     % Lower Plot (u radial) X-axis
     \put({20 + w_offset_lower}, {0 + h_offset_lower}){{1.1}}
     \put({43 + w_offset_lower}, {0 + h_offset_lower}){{1.2}}
     \put({65 + w_offset_lower}, {0 + h_offset_lower}){{1.3}}
     \put({87 + w_offset_lower}, {0 + h_offset_lower}){{1.4}}
     \put({10 + w_offset_lower}9, {0 + h_offset_lower}){{1.5}}
     \put({13 + w_offset_lower}1, {0 + h_offset_lower}){{1.6}}
     \put({15 + w_offset_lower}3, {0 + h_offset_lower}){{1.7}}
     \put({17 + w_offset_lower}5, {0 + h_offset_lower}){{1.8}}
     \put({19 + w_offset_lower}7, {0 + h_offset_lower}){{1.9}}

     % Lower Plot (u radial) Y-axis
     \put({11 + w_offset_lower},    {11 + h_offset_lower}) {{-0.8}}
     \put({11 + w_offset_lower},    {21.5 + h_offset_lower}) {{-0.6}}
     \put({11 + w_offset_lower},    {32 + h_offset_lower}) {{-0.4}}
     \put({11 + w_offset_lower},    {42.5 + h_offset_lower}) {{-0.2}}
     \put({13.25 + w_offset_lower}, {53.5 + h_offset_lower}) {{0.0}}
     \put({13.25 + w_offset_lower}, {64 + h_offset_lower}) {{0.2}}
     \put({13.25 + w_offset_lower}, {75 + h_offset_lower}) {{0.4}}
     \put({13.25 + w_offset_lower}, {86 + h_offset_lower}) {{0.6}}
     \put({13.25 + w_offset_lower}, {97 + h_offset_lower}) {{0.8}}

     % Upper Plot (u theta)
     \put(20, 145){{1.1}}
     \put(43, 145){{1.2}}
     \put(65, 145){{1.3}}
     \put(87, 145){{1.4}}
     \put(109, 145){{1.5}}
     \put(131, 145){{1.6}}
     \put(153, 145){{1.7}}
     \put(175, 145){{1.8}}
     \put(197, 145){{1.9}}

    % Upper Plot (u theta) Y-axis
     \put(11, 156) {{-0.8}}
     \put(11, 166.5) {{-0.6}}
     \put(11, 177) {{-0.4}}
     \put(11, 187.5) {{-0.2}}
     \put(13.25, 198.5) {{0.0}}
     \put(13.25, 209) {{0.2}}
     \put(13.25, 220) {{0.4}}
     \put(13.25, 231) {{0.6}}
     \put(13.25, 242) {{0.8}}
     
    % Upper Plot (u theta) Y-axis
     \put(11, {H + 11}) {{-0.8}}
     \put(11, {H + 21.5}) {{-0.6}}
     \put(11, {H + 32}) {{-0.4}}
     \put(11, {H + 42.5}) {{-0.2}}
     \put(13.25, {H + 53.5}) {{0.0}}
     \put(13.25, {H + 64}) {{0.2}}
     \put(13.25, {H + 75}) {{0.4}}
     \put(13.25, {H + 86}) {{0.6}}
     \put(13.25, {H + 97}) {{0.8}}

""")
