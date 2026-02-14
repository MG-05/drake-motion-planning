# drake-motion-planning
Sampling-based motion planning in Drake with collision avoidance in a shelf environment.

---

## Project Timeline

### Week 1 - Drake + Enviorment Setup

**Goal:**  

Set up the simulation environment that we can use a robot arm to plan in.

**Accomplishments:** 
- Installed and validated the Drake toolchain within the Python Environment.
- Built core Scene Environment:
  - Kuka iiwa manipulator model with claw.
  - Shelf Geometry to serve as an obstacle.
  - Brick Object for manipulation.
- Set up Project organization and to-dos for each iteration.

**Goals for Next Week:**  

Get the iiwa moving and be able to have the robot end-effector go from Point A to B.

**Media:**
<img width="1132" height="756" alt="scene_enviorment" src="https://github.com/user-attachments/assets/bf7c9862-b36a-4e61-9bd2-9eaa073f14cb" />

### Week 2 & 3 - Inverse Kinematics + Vanilla RRT 

**Goal:**  

Make the arm move reliably from a start end-effector configuration to a goal configuration 
without colliding with obstacles.

**Accomplishments:**
- Implemented an **Inverse Kinematics Pipeline** to map end-effector targets to to feasible joint configurations.
  - This was implemented through a **constrained optimization problem** via Drake's solver
- Implemented a **Vanilla RRT Planner** in the joint space of the robot to generate a collision-free path between two
feasible locations
- Integrated Inverse Kinematics with RRT so the user can identify the start and end configs, and the robot will find a
path if possible on its own.

**Challenges Overcome:**
- Inverse Kinematics Formulation:
  - As mentioned, the inverse kinematics problem is formulated as a constrained optimization problem in Drake. This is
  also a **non-convex problem** and therefore required us to make an initial guess which was the iiwa's start position.
  This was done so hopefully the IK find a feasible joint configuration near the original start joint configuration.
  
**Goals for Next Week:**  

The current pipeline with inverse kinematics and vanilla RRT is running into two major problems:
- In the configuration space, the shelf is captured as a narrow space. This naturally means that vanilla RRT will 
have a hard time finding a path if we place the goal end-effector too deep into a shelf. This is because RRT is a 
probabilistic sampler and sampling in tight regions is statically unlikely as compared to open regions. We need to fix
this with **RRT-Connect** which will run RRT in parallel from the start and end locations and hopefully have the two 
sampled trees meet.
- The second major problem is the jagged nature of the trajectory. This can be solved by doing **local trajectory
optimization**.  

The goals for next week are to fix these two problems.

**Media:**  
Lower Shelf Inverse Kinematics

https://github.com/user-attachments/assets/b29f0bf8-bba7-496c-9948-231a67df45c0

Middle Shelf Inverse Kinematics

https://github.com/user-attachments/assets/e5ccc8f2-0209-4959-8cad-8bd717a213bf

Upper Shelf Inverse Kinematics

https://github.com/user-attachments/assets/1a8a894c-0f86-4d84-9049-f5bb7f41dac4


### Week 4 - RRT-Connect

**Goal:**  

Make the arm move reliably from a start end-effector configuration to a goal configuration even when the goal config is
deep into the shelf

**Accomplishments:**
- Implemented **RRT-Connect** so every feasible config is reachable through probabilistic samplers. We can now reach into 
much deeper into the shelf.
- Implemented LCM to simulate within Drake. This allows the simulator to pick up on robot-robot and robot-enviornoment
collisions. Moreover, LCM simulation directly transfers to robot so simulating this is far beneficial for our long term 
goals of running the contact-rich manipulation on the a robot.

**Challenges Overcome:**
- LCM Collision Checker Tradeoffs
  - When running RRT-Connect we need a collision checker. The problem is that this collision checker checks against the
  environment and the robot itself. Checking against the robot is far too computational intensive causing the entire
  RRT-Connect to take about 2x more time. Fortunately, for our use-case we only are moving the final few joints allowing
  us to void the robot-robot checks. Therefore, we only do robot-enviorment checks allowing our RRT-Connect to converge
  much faster. The makes the average of 3000 iterations finish in a few seconds, as to a few minutes with robot-robot
  checks. 
  
**Goals for Next Week:**  

The next major goal is to get contract-rich manipulation and be able to move objects within the shelf. The next major
challenge we have is also to improve the RRT-Connect. The major problem we see is that since we keep the first connected 
tree in the RRT-Connect tree, we often get extraneous paths as shown below. The fix for this is to use a better and more
recent algorithm: Bi-RRT*.

https://github.com/user-attachments/assets/525f8280-9161-40b9-b4a1-a574fffa39f0

**Media:**  
Lower Shelf RRT-Connect

https://github.com/user-attachments/assets/63a17284-b79e-4822-9e18-5c4a511ddf62

Middle Shelf RRT-Connect

https://github.com/user-attachments/assets/9c16b3d8-a67f-4074-8d0c-383fa3877e28

Upper Shelf RRT-Connect

https://github.com/user-attachments/assets/ab93d09a-441c-4219-9d4c-3d20a5a6db17

