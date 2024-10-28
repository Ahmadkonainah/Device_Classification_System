package com.devicepricing;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Response class representing the prediction response from the Python API.
 */
@Data
@NoArgsConstructor  // Add this annotation
public class PredictionResponse {
    @JsonProperty("priceRange")
    private int priceRange;

    // Constructor
    public PredictionResponse(int priceRange) {
        this.priceRange = priceRange;
    }

    // Getters and setters
    public int getPriceRange() {
        return priceRange;
    }

    public void setPriceRange(int priceRange) {
        this.priceRange = priceRange;
    }
}
