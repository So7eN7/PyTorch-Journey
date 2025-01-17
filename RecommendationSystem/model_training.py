import torch
from model import RecommendationSystemModel
from data_preparation import le_user, le_movie
from device_setup import device
recommendation_model = RecommendationSystemModel(
    num_users=len(le_user.classes_),
    num_movies=len(le_movie.classes_),
    embedding_size=64,
    hidden_dim=128,
    dropout_rate=0.1,
).to(device)

optimizer = torch.optim.Adam(recommendation_model.parameters(), lr=1e-3)
loss_func = nn.MSELoss()