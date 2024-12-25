import numpy as np
from datetime import datetime
from collections import defaultdict
import pandas as pd


class RLShoeRecommender:
    def __init__(self, epsilon=0.1, learning_rate=0.1, discount_factor=0.95):
        self.epsilon = epsilon  # Za epsilon-greedy strategiju
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Čuvamo Q-vrednosti za svaki par (kontekst, proizvod)
        self.Q_values = defaultdict(lambda: defaultdict(float))

        # Čuvamo broj pokušaja za svaki par
        self.attempt_counts = defaultdict(lambda: defaultdict(int))

        # Istorija nagrada
        self.rewards_history = []

    def get_context(self, user_data, current_product=None):
        """
        Kreira kontekst na osnovu korisničkih podataka i trenutnog proizvoda
        """
        context_parts = []

        # Dodajemo demografske podatke ako postoje
        if "age_group" in user_data:
            context_parts.append(f"age_{user_data['age_group']}")
        if "gender" in user_data:
            context_parts.append(f"gender_{user_data['gender']}")

        # Dodajemo informacije o trenutnoj sesiji
        if "time_of_day" in user_data:
            context_parts.append(f"time_{user_data['time_of_day']}")
        if "device" in user_data:
            context_parts.append(f"device_{user_data['device']}")

        # Dodajemo informacije o trenutnom proizvodu
        if current_product:
            context_parts.append(f"category_{current_product['category_id']}")
            context_parts.append(f"brand_{current_product['brand_id']}")

        return "_".join(context_parts)

    def calculate_reward(self, action_taken, user_response):
        """
        Računa nagradu na osnovu akcije korisnika
        """
        rewards = {
            "purchase": 1.0,  # Kupovina
            "add_to_cart": 0.5,  # Dodavanje u korpu
            "click": 0.2,  # Klik na proizvod
            "view": 0.1,  # Pregled proizvoda
            "ignore": -0.1,  # Ignorisanje preporuke
            "bounce": -0.2,  # Napuštanje stranice
        }
        return rewards.get(user_response, 0.0)

    def select_action(self, context, available_products):
        """
        Bira proizvod za preporuku koristeći epsilon-greedy strategiju
        """
        if np.random.random() < self.epsilon:
            # Eksploracija: nasumični izbor
            return np.random.choice(available_products)

        # Eksploatacija: izbor najboljeg proizvoda
        q_values = {p: self.Q_values[context][p] for p in available_products}
        return max(q_values.items(), key=lambda x: x[1])[0]

    def update(self, context, action, reward, next_context=None):
        """
        Ažurira Q-vrednosti na osnovu dobijene nagrade
        """
        # Brojimo pokušaj
        self.attempt_counts[context][action] += 1

        # Računamo current Q-value
        current_q = self.Q_values[context][action]

        # Računamo next max Q-value ako postoji sledeći kontekst
        next_max_q = max(self.Q_values[next_context].values()) if next_context else 0

        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )

        # Ažuriramo Q-value
        self.Q_values[context][action] = new_q

        # Čuvamo nagradu u istoriji
        self.rewards_history.append(reward)

    def get_recommendations(
        self, user_data, current_product, available_products, n_recommendations=5
    ):
        """
        Vraća n preporuka za datog korisnika
        """
        context = self.get_context(user_data, current_product)
        recommendations = []

        # Biramo n proizvoda
        temp_products = available_products.copy()
        for _ in range(min(n_recommendations, len(available_products))):
            if not temp_products:
                break

            # Biramo proizvod
            product = self.select_action(context, temp_products)

            # Dodajemo u preporuke
            recommendations.append(product)

            # Uklanjamo iz dostupnih za sledeću iteraciju
            temp_products.remove(product)

        return recommendations

    def get_performance_metrics(self):
        """
        Vraća metrike performansi sistema
        """
        if not self.rewards_history:
            return None

        return {
            "average_reward": np.mean(self.rewards_history),
            "total_reward": np.sum(self.rewards_history),
            "reward_trend": pd.Series(self.rewards_history)
            .rolling(window=100)
            .mean()
            .tolist(),
            "total_attempts": sum(
                sum(counts.values()) for counts in self.attempt_counts.values()
            ),
        }


# Primer korišćenja:
if __name__ == "__main__":
    # Inicijalizacija recommendera
    recommender = RLShoeRecommender(epsilon=0.1)

    # Primer korisničkih podataka
    user_data = {
        "age_group": "25-34",
        "gender": "M",
        "time_of_day": "evening",
        "device": "mobile",
    }

    # Primer trenutnog proizvoda
    current_product = {"product_id": 1, "category_id": 1, "brand_id": 1}

    # Primer dostupnih proizvoda
    available_products = [1, 2, 3, 4, 5]

    # Dobijanje preporuka
    recommendations = recommender.get_recommendations(
        user_data, current_product, available_products, n_recommendations=3
    )

    # Simulacija korisničke akcije i ažuriranje sistema
    context = recommender.get_context(user_data, current_product)
    reward = recommender.calculate_reward(recommendations[0], "click")
    recommender.update(context, recommendations[0], reward)

    # Provera performansi
    metrics = recommender.get_performance_metrics()
    print("Preporuke:", recommendations)
    print("Metrike:", metrics)
