import os
import logging

try:
    from importlib.resources import as_file, files  # type: ignore
except ImportError:
    from importlib_resources import as_file, files  # type: ignore

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

fn = as_file(files("rpad_pybullet_envs_where2act_data").joinpath("."))
with fn as f:
    PARALLEL_JAW_GRIPPER_URDF = os.path.join(f, "panda_gripper.urdf")


class FloatingParallelJawGripper:
    def __init__(self, client_id):
        self.client_id = client_id

        # This is just a floating gripper yay.
        base_pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.base_id = p.loadURDF(
            PARALLEL_JAW_GRIPPER_URDF, *base_pose, physicsClientId=self.client_id
        )
        
        self.mass = 5
        
        self.activated = False
        self.contact_const = None
        self.contact_link_index = None
        
    def set_pose(self, pos, ori):
        p.resetBasePositionAndOrientation(self.base_id, pos, ori, self.client_id)
    
    def set_velocity(self, lin_vel, ang_vel):
         p.resetBaseVelocity(self.base_id, lin_vel, ang_vel, self.client_id)
         
    def detect_contact(self, obj_id):
        points = p.getContactPoints(bodyA=self.base_id, bodyB=obj_id, linkIndexA=0)
        return len(points) != 0
        
    
    def grasp(self, obj_id):
        if not self.activated:          
            contact_pts = p.getContactPoints(
                bodyA = self.base_id,
                bodyB = obj_id,
                linkIndexA = 0,
            )
            
            if contact_pts:
                contact_point = contact_pts[0]
                
                #gripper link
                gripper_link_pos, gripper_link_ori, _,_,_ = p.getLinkState(
                    bodyUniqueID=self.base_id,
                    linkIndex=0,
                    computeLinkVelocity=0,
                    physicsClientId=self.client_id
                )
                
                obj_link_pos, obj_link_ori, _, _, _, _ = p.getLinkState(
                    bodyUniqueId=obj_id,
                    linkIndex=contact_point[4],
                    computeLinkVelocity=0,
                    physicsClientId=self.client_id,
                )
                
                #transform contact pts in gripper link frame
                contact_pos_world = contact_point[5]
                contact_ori_world = gripper_link_ori
                contact_pos_gripper = p.invertTransform(
                    gripper_link_pos, gripper_link_ori, contact_pos_world
                )
                
                self.contact_const = p.createConstraint(
                    parentBodyUniqueId=self.base_id,
                    parentLinkIndex=0,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_point[4],
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0,0,0),
                    parentFramePosition=contact_pos_gripper,
                    parentFrameOrientation=gripper_link_ori,
                    childFramePosition=contact_point[6],
                    childFrameOrientation=obj_link_ori,
                    physicsClientId=self.client_id
                )
                
                self.activated=True
                self.contact_link_index = 0
    
    
    def apply_force(self, force):
        body_link_pos, body_link_ori, _, _, _, _ = p.getLinkState(
            bodyUniqueId=self.base_id,
            linkIndex=0,
            computeLinkVelocity=0,
            physicsClientId=self.client_id,
        )
        
        p.applyExternalForce(
            self.base_id,
            linkIndex=0,
            forceObj=force,
            posObj=body_link_pos,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client_id,
        )
    
    
    def release(self):
        if self.contact_const:
            p.removeConstraint(self.contact_const)
        self.activated = False
        self.contact_link_index = None
                
