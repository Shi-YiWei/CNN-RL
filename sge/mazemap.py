try:
    import pygame
    import pygame.freetype
except ImportError:
    pygame = None

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sge.utils import MOVE_ACTS, AGENT, BLOCK, WATER, KEY, OBJ_BIAS,\
    TYPE_PICKUP, TYPE_TRANSFORM, \
    WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED

CHR_WIDTH = 9
TABLE_ICON_SIZE = 40
MARGIN = 10
LEGEND_WIDTH = 250

__PATH__ = os.path.abspath(os.path.dirname(__file__))


class Mazemap(object):
    def __init__(self, game_name, game_config, render_config):
        # visualization
        if len(render_config)>0:
            self._rendering = render_config['vis']
            self.save_flag = render_config['save'] and self._rendering
            self.cheatsheet = render_config['key_cheatsheet'] and self._rendering
        else:
            self._rendering = False

        self.render_dir = './render'

        # load game config (outcome)
        self.gamename = game_name
        self.game_config = game_config
        if game_name == "playground":
            self.render_scale = 48
        elif game_name == 'mining':
            self.render_scale = 48
        else:
            raise ValueError("Unsupported : {}".format(game_name))
        self.table_scale = 32

        self.operation_list = self.game_config.operation_list
        self.legal_actions = self.game_config.legal_actions

        self.step_penalty = 0.0
        self.w = game_config.width
        self.h = game_config.height

        # map tensor
        self.obs = np.zeros((self.game_config.nb_obj_type + 3, self.w, self.h), dtype=np.uint8) #三维矩阵
        self.wall_mask = np.zeros((self.w, self.h), dtype=np.bool_)
        self.item_map = np.zeros((self.w, self.h), dtype=np.int16)

        self.init_screen_flag = False
        if self._rendering:

            print("self._rendering:",self._rendering)
            if pygame is None:
                raise ImportError("Rendering requires pygame installed on your environment: e.g. pip install pygame")
            self._init_pygame()
            self._load_game_asset()

    def reset(self, subtask_id_list):
        self.subtask_id_list = subtask_id_list
        self.nb_subtask = len(subtask_id_list)
        self.obs.fill(0)
        self.wall_mask.fill(0)
        self.item_map.fill(-1)
        self.empty_list = [] # record which position is empty

        self._add_blocks()
        self._add_targets()

    def act(self, action):
        oid = -1
        assert action in self.legal_actions, 'Illegal action: '
        if action in {KEY.UP, KEY.DOWN, KEY.LEFT, KEY.RIGHT}:  # move
            new_x = self.agent_x
            new_y = self.agent_y
            if action == KEY.RIGHT:
                new_x += 1
            elif action == KEY.LEFT:
                new_x -= 1
            elif action == KEY.DOWN:
                new_y += 1
            elif action == KEY.UP:
                new_y -= 1
            # wall_collision
            if not (new_x, new_y) in self.walls and not (new_x, new_y) in self.waters:
                self.obs[AGENT, self.agent_x, self.agent_y] = 0
                self.agent_x = new_x
                self.agent_y = new_y
                self.obs[AGENT, new_x, new_y] = 1
        else:  # perform
            iid = self._get_cur_item()
            if iid > -1:
                oid = iid-3
                self._perform(action, oid)  # perform action in the map
        self._process_obj()  # moving objects
        return oid

    def get_obs(self):
        return self.obs

    def _process_obj(self):
        for obj in self.object_list:
            oid = obj['oid']
            obj_param = self.game_config.object_param_list[oid]
            if 'speed' in obj_param and obj_param['speed'] > 0 and np.random.uniform() < obj_param['speed']:
                # randomly move
                x, y = obj['pos']
                candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                pool = []
                for nx, ny in candidates:
                    if self.item_map[nx, ny] == -1:
                        pool.append((nx, ny))
                if len(pool) > 0:
                    new_pos = tuple(np.random.permutation(pool)[0])
                    # remove and push
                    self._remove_item(obj)
                    self._add_item(oid, new_pos)

    def _remove_item(self, obj):
        oid = obj['oid']
        x, y = obj['pos']
        self.obs[oid+OBJ_BIAS, x, y] = 0
        self.item_map[x, y] = -1
        self.object_list.remove(obj)

    def _add_item(self, oid, pos):
        obj = dict(oid=oid, pos=pos)
        self.obs[oid+OBJ_BIAS, pos[0], pos[1]] = 1
        self.item_map[pos[0], pos[1]] = oid+OBJ_BIAS
        self.object_list.append(obj)

    def _perform(self, action, oid):
        assert(action not in MOVE_ACTS)
        act_type = self.operation_list[action]['oper_type']
        obj = None
        for oind in range(len(self.object_list)):
            o = self.object_list[oind]
            if o['pos'] == (self.agent_x, self.agent_y):
                obj = o
                break
        assert obj is not None

        # pickup
        if act_type == TYPE_PICKUP and self.game_config.object_param_list[oid]['pickable']:
            self._remove_item(obj)
        # transform
        elif act_type == TYPE_TRANSFORM and self.game_config.object_param_list[oid]['transformable']:
            self._remove_item(obj)
            outcome_oid = self.game_config.object_param_list[oid]['outcome']
            self._add_item(outcome_oid, (self.agent_x, self.agent_y))

    def _add_blocks(self):
        # boundary

        self.walls = [(0, y) for y in range(self.h)]  # left wall
        self.walls = self.walls + [(self.w-1, y) for y in range(self.h)]  # right wall
        self.walls = self.walls + [(x, 0) for x in range(self.w)]  # bottom wall
        self.walls = self.walls + [(x, self.h-1) for x in range(self.w)]  # top wall

        for x in range(self.w):
            for y in range(self.h):
                if (x, y) not in self.walls:
                    self.empty_list.append((x, y))
                else:
                    self.item_map[x, y] = 1  # block # item_map = 1 为有物体， -1 为空

        # random block
        if self.game_config.nb_block[0] < self.game_config.nb_block[1]:
            nb_block = np.random.randint(self.game_config.nb_block[0], self.game_config.nb_block[1]) #np.random.randint(1,3) = 2
            pool = np.random.permutation(self.empty_list)
            
            count = 0
            for (x, y) in pool:
                if count == nb_block:
                    break
                # based on self.item_map & empty_list
                if self._check_block(self.empty_list):

                    self.empty_list.remove((x, y))
                    self.walls.append((x, y))
                    self.item_map[x, y] = 1  # block
                    count += 1
            if count != nb_block:
                print('cannot generate a map without inaccessible regions! Decrease the #blocks')
                assert(False)
        for (x, y) in self.walls:
            self.obs[BLOCK, x, y] = 1 # obs block = 1

        # random water
        self.waters = []
        if self.game_config.nb_water[0] < self.game_config.nb_water[1]:
            nb_water = np.random.randint(
                self.game_config.nb_water[0], self.game_config.nb_water[1])
            pool = np.random.permutation(self.empty_list)
            count = 0
            for (x, y) in pool:
                if count == nb_water:
                    break
                if self._check_block(self.empty_list):  # success
                    self.empty_list.remove((x, y))
                    self.waters.append((x, y))
                    # water
                    self.item_map[x, y] = 2
                    self.obs[WATER, x, y] = 1
                    count += 1
            if count != nb_water:
                raise RuntimeError('Cannot generate a map without\
                    inaccessible regions! Decrease the #waters or #blocks')

    def _add_targets(self):
        # reset
        self.object_list = []
        self.omask = np.zeros((self.game_config.nb_obj_type), dtype=np.int8)

        # create objects
        # 1. create required objects
        pool = np.random.permutation(self.empty_list)
        for tind in range(self.nb_subtask):
            # make sure each subtask is executable
            self._place_object(tind, (pool[tind][0], pool[tind][1]))
        # 2. create additional objects
        index = self.nb_subtask
        for obj_param in self.game_config.object_param_list:
            if 'max' in obj_param:
                oid = obj_param['oid']
                nb_obj = np.random.randint(0, obj_param['max']+1)
                for i in range(nb_obj):
                    self._add_item(oid, (pool[index][0], pool[index][1]))
                    index += 1

        # create agent
        (self.agent_init_pos_x, self.agent_init_pos_y) = pool[index]
        self.agent_x = self.agent_init_pos_x
        self.agent_y = self.agent_init_pos_y

        self.obs[AGENT, self.agent_x, self.agent_y] = 1

    def _place_object(self, task_ind, pos):
        subid = self.subtask_id_list[task_ind]
        (_, oid) = self.game_config.subtask_param_list[subid]
        if ('unique' not in self.game_config.object_param_list[oid]) or \
            (not self.game_config.object_param_list[oid]['unique']) or \
                (self.omask[oid] == 0):
            self.omask[oid] = 1
            self._add_item(oid, pos)

    def _check_block(self, empty_list):
        nb_empty = len(empty_list)
        mask = np.copy(self.item_map)
        #
        queue = deque([empty_list[0]])
        x, y = empty_list[0]
        mask[x, y] = 1
        count = 0
        while len(queue) > 0:
            [x, y] = queue.popleft()
            count += 1
            candidate = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for item in candidate:
                if mask[item[0], item[1]] == -1:  # if empty
                    mask[item[0], item[1]] = 1
                    queue.append(item)
        return count == nb_empty

    def _get_cur_item(self):
        return self.item_map[self.agent_x, self.agent_y]
    # map

    # rendering
    def _init_pygame(self):
        self.title_height = 30
        pygame.init()
        pygame.freetype.init()

    #def _init_screen(self, graph_img, text_widths, num_lines):
    def _init_screen(self, text_widths, num_lines):

        obs_w = self.w*self.render_scale
        obs_h = self.h*self.render_scale #+ 30

        #graph_w, graph_h = graph_img.get_size()
        '''

        if self.cheatsheet:
            list_w = sum(text_widths)*CHR_WIDTH + TABLE_ICON_SIZE\
                    + MARGIN*(len(text_widths)-1)
            list_h = num_lines*TABLE_ICON_SIZE+10
        else:
            list_w, list_h = 0, 0
        '''
        list_w, list_h = 10, 40
        # obs_w + graph_w + list_w
        # size = [obs_w + graph_w + list_w + 45 + LEGEND_WIDTH, max(obs_h, list_h, graph_h) + self.title_height + 10]
        size = [ obs_w  + list_w , obs_h + list_h ]


        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("  ")

    def _load_game_asset(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        object_param_list = self.game_config.object_param_list
        self.object_image_list = []
        self.obj_img_plt_list = []
        img_folder = os.path.join(ROOT_DIR, 'asset', self.gamename, 'Icon')
        for obj in object_param_list:
            image = pygame.image.load(os.path.join(img_folder, obj['imgname']))
            self.object_image_list.append(image)
            image = plt.imread(os.path.join(img_folder, obj['imgname']))
            self.obj_img_plt_list.append(image)
        self.agent_img = pygame.image.load(os.path.join(img_folder, 'agent.png'))

        if self.gamename == 'mining':
            self.block_img = pygame.image.load(os.path.join(img_folder, 'mountain.png'))
            self.water_img = pygame.image.load(os.path.join(img_folder, 'water.png'))
        else:
            self.block_img = pygame.image.load(os.path.join(img_folder, 'block.png'))

    def render(self, step_count, text_lines, text_widths, status, bg_colors):
        if not self._rendering:
            return
        pygame.event.pump()
        GAME_FONT = pygame.freetype.SysFont('Arial', 20)
        STAT_FONT = pygame.freetype.SysFont('Arial', 24)
        TITLE_FONT = pygame.freetype.SysFont('Arial', 30)

        #print("__PATH__:"+__PATH__)
        # comment
        # graph_img = pygame.image.load(os.path.join(__PATH__, '../render/temp/subtask_graph.png'))
        # comment
        if not self.init_screen_flag:
            #self._init_screen(graph_img, text_widths, len(text_lines))
            self._init_screen(text_widths, len(text_lines))
            self.init_screen_flag = True
            self.arrow_img = pygame.image.load(os.path.join(__PATH__, 'asset/arrow.png'))

        self.screen.fill(WHITE)
        size = [self.render_scale, self.render_scale]
        w_bias, h_bias = 0, 0
        tbias = self.title_height

        # 1. Observation
        # items
        for i in range(len(self.object_list)):
            obj = self.object_list[i]
            oid = obj['oid']
            obj_img = self.object_image_list[oid]
            if obj_img.get_width() != self.render_scale:
                obj_img = pygame.transform.scale(obj_img, size)
            self.screen.blit( obj_img, (obj['pos'][0]*self.render_scale, tbias+obj['pos'][1]*self.render_scale))
        # walls
        if self.block_img.get_width() != self.render_scale:
            self.block_img = pygame.transform.scale(self.block_img, size)
        for wall_pos in self.walls:
            pos = [self.render_scale * wall_pos[0],
                   tbias+self.render_scale * wall_pos[1]]
            self.screen.blit(self.block_img, pos)
        # waters
        if self.gamename == 'mining':
            if self.water_img.get_width() != self.render_scale:
                self.water_img = pygame.transform.scale(self.water_img, size)
            for water_pos in self.waters:
                pos = [self.render_scale * water_pos[0], tbias+self.render_scale * water_pos[1]]
                self.screen.blit(self.water_img, pos)
        # agent
        if self.agent_img.get_width() != self.render_scale:
            self.agent_img = pygame.transform.scale(self.agent_img, size)
        pos = (self.render_scale * self.agent_x, tbias+self.render_scale * self.agent_y)
        self.screen.blit(self.agent_img, pos)
        # grid
        for x in range(self.w+1):
            pygame.draw.line(self.screen, DARK, [x*self.render_scale, tbias], [x*self.render_scale, tbias+self.h*self.render_scale], 3)
        for y in range(self.h+1):
            pygame.draw.line(self.screen, DARK, [0, tbias+y*self.render_scale], [self.w*self.render_scale, tbias+y*self.render_scale], 3)

        title_x = round(self.w*self.render_scale/2)-80
    
        # draw
        pygame.display.flip()

        ''' save image
        ===== ===== ===== ===== 
        if self.save_flag:
            self._save_image(step_count)
        ===== ===== ===== =====
        '''

        # ===== ===== ===== ===== new code ===== ===== ===== =====

        screen_numpy = pygame.surfarray.array3d(self.screen)
        screen_numpy_new = np.transpose(screen_numpy, ( 2, 0, 1))

        return screen_numpy_new

        # ===== ===== ===== ===== new code ===== ===== ===== =====

    def _add_box_with_label(self, x, y, W, H, label, font, color):
        pygame.draw.rect(self.screen, color, (x, y, W, H), 0)
        pygame.draw.rect(self.screen, BLACK, (x, y, W, H), 2)
        font.render_to(self.screen, (x + 5, y + 5), label)







        

    def _save_image(self, step_count):
        if self._rendering and self.render_dir is not None:

            screen_numpy = pygame.surfarray.array3d(self.screen)

            #print("type screen_numpy:", type(screen_numpy))
            #print("shape screen_numpy:", screen_numpy.shape)

            screen_numpy_new = np.transpose(screen_numpy, ( 2, 0, 1))
            #print("type screen_numpy:", type(screen_numpy_new))
            #print("shape screen_numpy:", screen_numpy_new.shape)



            # print(type(screen))
            pygame.image.save(self.screen, self.render_dir + '/render' + '{:02d}'.format(step_count) + '.jpg')
        else:
            raise ValueError('_rendering is False and/or environment has not been reset')
