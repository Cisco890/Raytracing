import struct

class BMPTexture(object):
    def __init__(self, filename):
        with open(filename, "rb") as image:
            image.seek(10)
            headerSize = struct.unpack('=l', image.read(4))[0]
            image.seek(18)
            self.width = struct.unpack('=l', image.read(4))[0]
            self.height = struct.unpack('=l', image.read(4))[0]
            image.seek(headerSize)
            self.pixels = []
            for y in range(self.height):
                pixelRow = []
                for x in range(self.width):
                    b = ord(image.read(1)) / 255
                    g = ord(image.read(1)) / 255
                    r = ord(image.read(1)) / 255
                    pixelRow.append([r,g,b])
                self.pixels.append(pixelRow)
    
    def getColor(self, u, v):
        # Clamp UV coordinates to [0, 1]
        u = max(0, min(1, u))
        v = max(0, min(1, v))
        
        # Convert to pixel coordinates
        x = int(u * (self.width - 1))
        y = int(v * (self.height - 1))
        
        # Ensure we're within bounds
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        
        return self.pixels[y][x]
