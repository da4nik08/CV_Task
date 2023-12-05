from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image, ImageDraw
import os
import glob


def Get_coords(centr, labels):

    
    def Get_true(centr, labels, alpha):
        mins = []
        for c in centr:
            lst = []
            for c2 in centr:
                if np.array_equal(c, c2): continue 
                dist = np.linalg.norm(c - c2)
                lst.append(dist)
            mins.append(min(lst))
    
        median = sorted(mins)[len(mins)//2]
    
        filtr_idx = []
        for idx, m in enumerate(mins):
            if m/median > alpha:
                filtr_idx.append(idx)
                
        filtr_centr = np.delete(centr, filtr_idx, axis=0)
        filtr_labels = np.delete(labels, filtr_idx)
        
        return filtr_centr, filtr_labels

    
    def map_coordinates_tocells(coordinates, labels, min_y, min_x, cell_height, cell_width):
        # Initialize a 3x3 grid to represent the Tic Tac Toe board
        grid = [[' ' for _ in range(3)] for _ in range(3)]
        xy = [[' ' for _ in range(3)] for _ in range(3)]
        # Iterate through the coordinates and labels
        for (x, y), label in zip(coordinates, labels):
            # Calculate the row and column based on the coordinates
            row = round((y - min_y) / cell_height)
            col = round((x - min_x) / cell_width)
            # Update the grid with the label (X or O)
            grid[row][col] = label
            xy[row][col] = [x, y]
    
        return grid, xy


    def find_winner_line(grid):
        # Check rows
        for row in range(3):
            if grid[row][0] == grid[row][1] == grid[row][2] and grid[row][0] != " ":
                return [(row, 0), (row, 1), (row, 2)]
    
        # Check columns
        for col in range(3):
            if grid[0][col] == grid[1][col] == grid[2][col] and grid[0][col] != " ":
                return [(0, col), (1, col), (2, col)]
    
        # Check diagonals
        if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != " ":
            return [(0, 0), (1, 1), (2, 2)]
        if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != " ":
            return [(0, 2), (1, 1), (2, 0)]
    
        return None  # No winner line found


    c, l = Get_true(centr, labels, 2)
    
    xmax, ymax = np.max(c, axis=0)
    xmin, ymin = np.min(c, axis=0)

    cell_height = (ymax - ymin) / 2
    cell_width = (xmax - xmin) / 2

    grid, xy = map_coordinates_tocells(c, l, ymin, xmin, cell_height, cell_width)

    win = find_winner_line(grid)

    if win == None: return None, None

    xy1 = xy[win[0][0]][win[0][1]]
    xy2 = xy[win[2][0]][win[2][1]]
    return xy1, xy2


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YOLO("best.pt")
    
    folder_path = os.path.join(os.getcwd(),'imgs')
    output_path = os.path.join(os.getcwd(),'out_imgs')
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']

    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
    
    for img_path in image_files:
        img = Image.open(img_path)
    
        results = model(img, conf=0.4, iou=0.1)
        
        for r in results:
            labels = r.boxes.cls.cpu().numpy()
            coordinates = r.boxes.xywh.cpu().numpy()
            
        centr = coordinates[:,0:2]
        
        xy1, xy2 = Get_coords(centr, labels)
        
        draw = ImageDraw.Draw(img)
        print(xy1, xy2)
        if xy1 != None:
            draw.line([xy1[0], xy1[1], xy2[0], xy2[1]], fill="black", width=8)
        img.save(os.path.join(output_path, img_path.split("\\")[-1]))
        
    
    
main()
