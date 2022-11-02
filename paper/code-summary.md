features:


jargon:
mission: To complete the LTL statement while traversing through the enviroment
axion: a statement/process that can output true or false with an input of a cell
full_image_plan: plan everything including creating img_cells from scratch
full_replan: replan the DJK/A* part of plan, the img_cells are updated as they are seen
partial_replan: replan only a section of the DJK/A*

current method:
read and process image
- read image of our environment
- perspective warp and crop image
- mask and extract the image into high rgby channels
- merge the masked image back into a single color (processed_img)
- merge just all the channels except green to get the combined reward image (raw_reward_image)
- return the processed_image, raw_reward_image and the raw risk vales (green_channel)

create reward images
- take our processed_img and split the environment into cells, return the cell typed (cell_type) as 2D array **CHECK THIS**[y, x] and the accumulated risk calue for that cell as a similar 2D aray (cell_cost), and our processed_img with the cell boundries drawn in (img_cells)
- convert the goal and objective cells into individual and named goals and objectives
- get the mission start and finish locations (global_start, global_finish) (collectively: mission_phy_bounds)
- calculate the reward images (reward_graphs) and place them in a dictionary {'axion' : <reward_image>}
- return img_cells, reward_graphs, mission_phy_bounds

parse ltl formula automata
- read in file
- process file to create and return ltl_state_diag, aps, start_ltl_state, final_ltl_state
get our base assumed risk "satellite image"
- create our risk image

pathfind (different path finding methods are used and discribed at end)
- todo

output the path
- split image into rgb channels
- add raw_risk_image to all channels, this will make the actual risk items white, we do this because raw risk blends into the discovered/assumed risk image
- merge back the rgb channels (dj_path_image)
- create our cells on dj_path_image to make it look pretty (dj_path_image)
- draw out path on the dj_path_image
- return dj_path_image




pathfind_updating_risk:
- get start ltl and start phys state
- create an array to store the path we have taken
- create a copy of our assumed risk and make a copy we can fill out
- create variable (img_tmp_idx_ltl) so our pictures have unique names and they are ordered
- start the ltl traversal loop
    - get the current reward images for all the possible paths out of this current node (current_ltl_state_reward_graph)
    - create empty variable for our final phys location for this ltl node (next_phys_loc)
    - create variable (img_tmp_idx_phys) so our pictures have unique names and they are ordered
    - create our empty shortest path variable so we can skip planning if the risk hasnt changed. we can just go to the next cell after current_phys_loc
    - create our pathfinding data structures, the structures must be in this scope so we can partial plan
    - start the physical path planning loop
        - insert out curent location into the cells we have visited (**CHECK THIS**should this go at the end of the loop)
        - update our assumed risk map filled map with risk values from our viewing distance and get the cells that changed
        - create variable for the astar target for partial plans
        - if we havent planned ever, do a full_image_plan
            - merge our current_ltl_state_reward_graph with our assumed_risk_image_filled (risk_reward_image_local) to create our enviroment image
            - create our cell image (risk_reward_img_cells_local), cells types (risk_reward_cell_type_local), and cell cost (risk_reward_cell_cost_local) on   risk_reward_image_local
            - convert risk_reward_cell_type_local and risk_reward_cell_cost_local into a state diagram
            - get out next_phys_loc based off of risk_reward_cell_type_local
            - calculate djk/astar and store it in shortest_path
        - if we have already done full_image_plan, do a quicker replan instead
            - merge our current_ltl_state_reward_graph with our assumed_risk_image_filled (risk_reward_image_local) to create our enviroment image
            - update only the cells that have risk updates in our cell map rather than creating it from scratch
            - if the amount of risk updated pixels are greater than 0 (we updated our assumed_risk_image_filled map) then partial_replan
                - convert our cell type and cell cost into a state_diagram
                - calculate the partial plan target astar_target
                - calculate djk/astar from current loc to partial plan target and store it in shortest_path_astar_target
                - splice the shorter astar target path into shortest_path (this is the current plan)
        - if debugging output image
        - increment current_phy_loc to the next loc in shortest_plan
    - end "physical path planning loop"
    - get the next ltl state we need to go to based off the current location (should be next_phys_loc)
- end "ltl traversal loop"
- return path taken and assumed_risk_image_filled




pathfind_no_sensing_range:
- get start ltl and start phys state
- create an array to store the path we have taken
- create variable (img_tmp_idx_ltl) so our pictures have unique names and they are ordered (TODO)
- start the ltl traversal loop
    - get the current reward images for all the possible paths out of this current node (current_ltl_state_reward_graph)
    - merge our current_ltl_state_reward_graph with our assumed_risk_image_filled (risk_reward_image) to create our enviroment image
    - create our cell image (risk_reward_img_cells), cells types (risk_reward_cell_type), and cell cost (risk_reward_cell_cost) on   risk_reward_image
    - convert risk_reward_cell_type and risk_reward_cell_cost into a state diagram
    - get out next_phys_loc based off of risk_reward_cell_type
    - calculate djk/astar and store it in shortest_path
    - splice shortest_path into total_shortest_path
    - get the next ltl state we need to go to based off the current location (should be next_phys_loc)
- end "ltl traversal loop"
- return path taken and assumed_risk_image_filled (should be unchanged since we dont have a sening region)




pathfind_product_auto: basic path finding
- create state diagram
- create product automata with state_diagram and ltl_state_diag
- astar the product automata
- travel entire path




pathfind_product_sensing_region: basic path finding with a sensing region
- update risk
- create required data structures
- create state diagram
- create product automata with state_diagram and ltl_state_diag
- astar the product automata
- travel one cell
- maybe only update product automata cells that have changed
