
import cv2
import numpy as np
from skimage.metrics import structural_similarity
import random
import settings

def scan(image):

    # #TODO Check that the edges are paper after scanning

    def four_point_transform(image, pts):
        # Obtain a consistent order of the points
        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute the width of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Compute the perspective transform matrix and apply it
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def order_points(pts):
        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    orig = image.copy()
    ratio = image.shape[0] / 500.0

    # Preprocess the image
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 75, 200)

    # Find contours and keep the largest one
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If our approximated contour has four points, we can assume we have found the paper
        if len(approx) == 4:
            screenCnt = approx
            break

    # Apply the perspective transform
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # Convert the warped image to grayscale
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Return the scanned image

    return resize(warped)

def black_white(image, bias = -12):
    # Bias to make it more white (negative bias)
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding

    threshold_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    biased_threshold = int(threshold_value) + int(bias)
    biased_threshold = max(0, min(biased_threshold, 255))

    _, bw = cv2.threshold(blurred, biased_threshold, 255, cv2.THRESH_BINARY)


    return bw

def difference(before, after, threshold = 1000):
    # Convert images to grayscale

    if len(before.shape) == 3:
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    else:
        before_gray = before

    if len(after.shape) == 3:
        
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY, )
    else:
        after_gray = after

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > threshold:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(c)


            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            
            # Put -1 as last parameter to fill the rectangle 
            cv2.drawContours(mask, [c], 0, (255, 255, 255), 2)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), 2)  # Change thickness as needed

    return mask

def resize(img):
    # Make image landscape
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    # Resize image to 640x480
    img = cv2.resize(img, (640, 480))

    return img

def check_curve_closed(image):

    black_white_image = black_white(image)

    # Set the fill color and a seed point
    fill_color = 255*3
    seed_point = (random.randint(0, black_white_image.shape[1]-1), random.randint(0, image.shape[0]-1))

    # Copy the original image to perform the flood fill
    floodfilled_image = black_white_image.copy()

    # Create a mask for flood fill function (1 pixel border for the mask)
    h, w = floodfilled_image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Perform the flood fill
    cv2.floodFill(floodfilled_image, mask, seed_point, fill_color)

    # Check if any non-white (non-filled) pixel exists, which indicates a closed curve
    if np.all(floodfilled_image == 255):
        return False, 0

    # Save the image for debugging
    # cv2.imwrite('floodfilled_image.jpg', floodfilled_image)

    # Save all black pixels
    black_region = np.where((floodfilled_image == 0))

    # Count the number of filled pixels
    filled_area = np.sum(floodfilled_image == 0)

    # Calculate the ratio of filled area to total area
    total_area = h * w
    filled_area_ratio = filled_area / total_area

    return black_region, filled_area_ratio, True

def check_parallel_lines(path, image):
    # Retrieve settings
    width = settings.MAP_WIDTH
    height = settings.MAP_HEIGHT

    # Margin
    m = 10 

    # Points for flood fill
    point_1 = (m, m)
    point_2 = (width - m, height - m)

    # Convert image to black and white
    black_white_region = black_white(image)

   # Create a color copy of the image to apply flood fill
    color_region = cv2.cvtColor(black_white_region, cv2.COLOR_GRAY2BGR)

    # Create masks for flood fill (1 pixel border for the mask)
    h, w = black_white_region.shape[:2]
    mask_1 = np.zeros((h+2, w+2), np.uint8)
    mask_2 = np.zeros((h+2, w+2), np.uint8)

    # Colors for flood fill (red and blue)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)

    # Perform flood fill from point_1 with red color
    cv2.floodFill(color_region, mask_1, point_1, color_red)

    # Perform flood fill from point_2 with blue color
    cv2.floodFill(color_region, mask_2, point_2, color_blue)

    # Count how many colors are in the image
    unique_colors = np.unique(color_region.reshape(-1, color_region.shape[2]), axis=0)
    num_colors = len(unique_colors)
    
    if num_colors != 4: # red + blue + black + white
        raise Exception("The lines do not split the region into 3 parts")
    if path == 'top-bottom':
        # Check if the first and last rows are the same color
        if np.array_equal(color_region[0], color_region[-1]):
            return False
    elif path == 'left-right':
        # Check if the first and last columns are the same color
        if np.array_equal(color_region[:,0], color_region[:,-1]):
            return False
    elif path == 'curved':
        # Check if the first and last pixels are not the same color
        if not np.array_equal(color_region[0,0], color_region[-1,-1]):
            return False
    else:
        raise Exception("Invalid path")

    # Take all white and black pixels
    black_white_region = np.where((color_region == (255, 255, 255)).all(axis=2), 255, 0)

    # Area of the black and white region to total area ratio
    total_area = h * w
    filled_area = np.sum(black_white_region == 0)
    filled_area_ratio = filled_area / total_area


    # Save the image for debugging
    cv2.imwrite('color_region.jpg', color_region)

    return black_white_region, filled_area_ratio, True

def check_region(image, region):

    if region.name == 'river' or region.name == 'road':
        region_check = check_parallel_lines(region.path, image);
    elif region.name == 'forest' or region.name == 'bush':
        region_check = check_curve_closed(image);
    
    if region_check[2] == False:
        return False

    if region_check[1] < region.min_instace_size:
        raise Exception(f"The {region.name} is too small")
    elif region_check[1] > region.max_instance_size:
        raise Exception("The {region.name} is too big")
    
    return region_check[0], True


def save_to_map(map, pixels, region):
    code = region.code

    for pixel in pixels:
        if map[pixel[0], pixel[1]] != 'g' or map[pixel[0], pixel[1]] != ' ':
            raise Exception("The region overlaps with another region")
        else:
            map[pixel[0], pixel[1]] = code

# diff = difference(img_1, img_2)

diff = cv2.imread('diff_test.jpg')
print(check_parallel_lines(diff))    


