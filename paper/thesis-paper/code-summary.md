
Typically in a pathfinding algorithm with tasks, the enviroment is broken into cells and converted into a automata. The states are the cells the enviroment is made op of and each cell has a 8 transitions, one going up to the cell above it, one coming down from that cell. Each other direction has two transitions, one going into the current cell and one going out. The task the pathfinding robot must compleate is can be represented by an LTL equation, this equation can be converted into a automata that defined the valid transions into the various states the task can exhibit.

A typical algorithm would calculate the product of these two automata and then apply a graph search algorithm to find a path from the start to the end. The complexety of this processes is min m x n, where m is the nodes/cells in the enviroment, and n is the number of nodes in the LTL equation converted into the automata. In large engiroments and large task space. This can intoduce many inneffiencies. If a corner of the enviroment is never traveled to, there is no need to calculate the paroduct in this area. This arguement can also be used to reduce the number of nodes in the LTL automata to reduce the complexity. By creating an algorithm that calculates the product automata on the fly, many of these inefficiencies can be eliminated by a marginal drop in perfoemance (performance being the absulute shortest path from start to finish). This balence between speed and performance is a long standing question that has been puzzling many academic. In this paper, we will see that for small envirment and small task space, a conventional product automata may be beneficial, but for large enviroments this may offer many benefits such as, partially known enviroments, unknown enviroments, risky envirments, changing tasks, multi-agent systems and more.


three examples
- street pathfinding example (don't know edges of road well)
- military example (circle example) (ai)
- lunar example


# Enviroment:
An enviroment is defined by an image with two areas. The first area is representative of risk value. The risk value is represented from a scale of 0 to 255 where 255 is the max risk possible. The other areas are the target areas. These areas are the task locations for the LTL formula.

[enviroment raw](/paper/env1.bmp)

The risk is then concealed using a bluring to similate a partially known enviroment. This process is optional.

[pk enviroment](/paper/env2.bmp)

As the agent moves about the enviroment uncovering the actual risk value and updating the path it should take.

# LTL
INSERT LTL COPYPASTA (BUT WITH MY WORDS) STUFF HERE

# Risk

# Algorithm
## Product Automata
The typical algorithm for a LTL based pathfinding is to convert the LTL into a buchii automata. The enviroment is split into cells and converted into a graph. Each cell has 8 edges. 4 for each cardinal direction out of the cell and 4 more for directions into the cell. Both of these objects are graphs so a product of these two can create a larger graph with all the potential states and transitions. A djikstras algorithm can bb used on this larger graph to find the path through the envoriment while compleating the LTL task.

## risk aware planning
One of

## chunk live planning
Using a live algorithm that calulates the product automata on the fly can help speed up many calculations and prevent valuable computer cycles from being used in unnessicary computations. This allows the algoithm to incorporate live data from the enviroment or a changes from a human manager to more efficiently plan a new path.


pathfind():
    heuristically caluate best target
    use astar to pathfind to target

the best target can be calulated with many heuristical functions. A few different heuristical functiosn are discussed in this paper.

heuristic 1: (LTL djk)
By utilizing djikstras algoritm to weight all the nodes in the ltl buchii automata, a very simple, albeit inifficient heuristic can be utilized. Since this automata can be viewed as a graph, applying djk's will lead to the shortest path on the LTL landscape graph.

heuristic 2: (euclidian)

heiristic 3:

The algorithm can now minimize the cost function primarly on the LTL landscape then the minimize the cells traveled in the physical landscape.

## live enviroments
By calculating the product automata on-the-fly, the enviroment can change, update and more information from the enviroment can be extracted. If the enviroment were to change, the path could be very quickly recalculated without having to do the complecated and potentially

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
- see algorithms below

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
