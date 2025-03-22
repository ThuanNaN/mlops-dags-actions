import psycopg2
from PIL import Image
import torch
from io import BytesIO
from torchvision import transforms
from PIL import Image
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])

db_params = {
    "host": "localhost",     
    "database": "image_db",   
    "user": "postgres",      
    "password": "aivn2025",  
    "port": "5454"     
}

def binary_to_image(binary_data):
    buffer = BytesIO(binary_data)
    return Image.open(buffer)

def load_image_from_db(table_name: str):
    connection = None
    cursor = None
    original_images = []
    grayscale_images = []
    try:
        connection = psycopg2.connect(**db_params)
        print("Connected to the database successfully!")
        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT * FROM {table_name}
        """)
        rows = cursor.fetchall()

        if rows:
            for idx, row in enumerate(rows):
                _, _, _, original_data, grayscale_data, _  = row
                original_array = np.frombuffer(original_data, dtype=np.uint8).reshape(224, 224, 3)
                grayscale_array = np.frombuffer(grayscale_data, dtype=np.uint8).reshape(224, 224)

                original_images.append(original_array)
                grayscale_images.append(grayscale_array)
        else:
            print("No data found!")
    except psycopg2.Error as e:
        print(f"Error: Could not connect to the database: {e}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Connection closed.")
        return original_images, grayscale_images


class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, data_name: str):
        self.original_images, self.grayscale_images = load_image_from_db(data_name)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        original_image = np.copy(self.original_images[index])  
        grayscale_image = np.copy(self.grayscale_images[index])  
        
        original_image = self.transform(original_image)
        grayscale_image = self.transform(grayscale_image)
        return original_image, grayscale_image

    def __len__(self):
        return len(self.original_images)


if __name__ == "__main__":
    dataset = ColorizationDataset("source_images")
    print(f"Dataset length: {len(dataset)}")
    original_image, grayscale_image = dataset[0]
    print(f"Original image shape: {original_image.shape}")
    print(f"Grayscale image shape: {grayscale_image.shape}")
    print("Data loaded successfully!")