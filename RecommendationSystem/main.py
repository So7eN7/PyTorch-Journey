# Import necessary modules
from device_setup import device
from data_loading import df
from data_visualization import *
from data_preparation import train_dataset, valid_dataset, le_user, le_movie
from dataloader import train_loader, val_loader
from model_training import recommendation_model, optimizer, loss_func
from training_loop import EPOCHS, log_progress, losses
from training_visualization import *
from evaluation import rms
from precision_recall import user_precisions, user_based_recalls, k, threshold
from precision_recall_metrics import average_precision, average_recall
from recommendations import recommend_top_movies, get_movies_with_genres, df_movies, all_movies, seen_movies, user_id, recommendations, recommended_movies_with_genres, seen_movies_with_genres
# Main function to run the entire workflow
def main():
    # Device setup
    print(device)

    # Data loading and visualization
    df.head()
    df.describe()
    df.isnull().sum()
    print(df.userId.nunique(), df.movieId.nunique())
    print(df.rating.value_counts())
    plt.figure(figsize=(8, 6))
    plt.hist(df.rating)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Count of Ratings")
    plt.show()

    # Data preparation
    train_dataset_size = len(train_dataset)
    print(f"Training on {train_dataset_size} samples...")

    # Model training
    recommendation_model.train()
    for e in range(EPOCHS):
        step_count = 0  # Reset step count at the beginning of each epoch
        for i, train_data in enumerate(train_loader):
            output = recommendation_model(
                train_data["users"].to(device), train_data["movies"].to(device)
            )
            # Reshape the model output to match the target's shape
            output = output.squeeze()  # Removes the singleton dimension
            ratings = (
                train_data["ratings"].to(torch.float32).to(device)
            )  # Assuming ratings is already 1D

            loss = loss_func(output, ratings)
            total_loss += loss.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Increment step count by the actual size of the batch
            step_count += len(train_data["users"])

            # Check if it's time to log progress
            if (
                step_count % log_progress_step == 0 or i == len(train_loader) - 1
            ):  # Log at the end of each epoch
                log_progress(
                    e, step_count, total_loss, log_progress_step, train_dataset_size, losses
                )
                total_loss = 0

    # Training visualization
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Steps")
    plt.ylabel("Avg Loss")
    plt.show()

    # Evaluation
    recommendation_model.eval()
    with torch.no_grad():
        for i, valid_data in enumerate(val_loader):
            output = recommendation_model(
                valid_data["users"].to(device), valid_data["movies"].to(device)
            )
            ratings = valid_data["ratings"].to(device)
            y_pred.extend(output.cpu().numpy())
            y_true.extend(ratings.cpu().numpy())

    # Calculate RMSE
    rms = mean_squared_error(y_true, y_pred, squared=False)
    print(f"RMSE: {rms:.4f}")

    # Precision and Recall
    with torch.no_grad():
        for valid_data in val_loader:
            users = valid_data["users"].to(device)
            movies = valid_data["movies"].to(device)
            ratings = valid_data["ratings"].to(device)
            output = recommendation_model(users, movies)

            for user, pred, true in zip(users, output, ratings):
                user_ratings_comparison[user.item()].append((pred[0].item(), true.item()))

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        user_precisions[user_id] = precision
        user_based_recalls[user_id] = recall

    average_precision = sum(prec for prec in user_precisions.values()) / len(
        user_precisions
    )
    average_recall = sum(rec for rec in user_based_recalls.values()) / len(
        user_based_recalls
    )

    print(f"precision @ {k}: {average_precision:.4f}")
    print(f"recall @ {k}: {average_recall:.4f}")

    # Recommendations
    recommendations = recommend_top_movies(
        recommendation_model, user_id, all_movies, seen_movies, device
    )

    recommended_movies_with_genres = get_movies_with_genres(recommendations, df_movies)
    user_top_ten_seen_movies = df[df['userId'] == user_id].sort_values(by="rating", ascending=False).head(10)
    seen_movies_with_genres = get_movies_with_genres(user_top_ten_seen_movies['movieId'], df_movies)

    print(f"Recommended movies:\n\n{recommended_movies_with_genres}\n\nbased on these movies the user has watched:\n\n{seen_movies_with_genres}")

if __name__ == "__main__":
    main()
