import cv2
import numpy as np
from datetime import datetime
import argparse


def runstamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def get_points_from_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100,150,0), (140,255,255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w//2
        cy = y + h//2
        points.append(np.array([cx, cy], dtype=float))

    points.sort(key=lambda p: (p[0], p[1]))
    return points


def dist(a, b):
    return np.linalg.norm(a - b)


def nearest_neighbor_route(start_index, points):
    unvisited = list(range(len(points)))
    route = [start_index]
    unvisited.remove(start_index)
    current = start_index

    while unvisited:
        next_point = min(unvisited, key=lambda idx: dist(points[current], points[idx]))
        route.append(next_point)
        unvisited.remove(next_point)
        current = next_point

    return route


def route_length(route, points):
    diffs = [points[route[i]] - points[route[i+1]] for i in range(len(route)-1)]
    return sum(np.linalg.norm(d) for d in diffs)


def draw_route(img, points, route, outfile):
    for i in range(len(route) - 1):
        p1 = tuple(points[route[i]].astype(int))
        p2 = tuple(points[route[i+1]].astype(int))
        cv2.line(img, p1, p2, (0, 255, 0), 2)

    cv2.imwrite(outfile, img)


def main():
    RUNSTAMP = runstamp()
    outfile = f"nearest_neighbor_path-{RUNSTAMP}.png"

    parser = argparse.ArgumentParser(description="Compute nearest‑neighbor path through colored points.")
    parser.add_argument(
        "inputfile",
        nargs="?",
        default="map.png",
        help="Input image filename (default: map.png)"
    )
    args = parser.parse_args()

    img = cv2.imread(args.inputfile)
    if img is None:
        print(f"ERROR: Could not read image '{args.inputfile}'")
        return

    points = get_points_from_image(img)

    # Compute all nearest neighbor routes
    all_routes = []
    for i in range(len(points)):
        r = nearest_neighbor_route(i, points)
        length = route_length(r, points)
        all_routes.append((i, r, length))

    # Pick the best nearest neighbor route
    best_start, best_route, best_len = min(all_routes, key=lambda x: x[2])
    print(f"Best start index: {best_start}, route length: {best_len}")

    # Generate Output and save the image with the drawn route
    draw_route(img, points, best_route, outfile)
    print(f"Wrote: {outfile}")


if __name__ == "__main__":
    main()
