package com.devicepricing;

import java.time.Duration;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.stream.Collectors;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestTemplate;

import lombok.extern.slf4j.Slf4j;

/**
 * Configuration for RestTemplate
 * Sets up the RestTemplate bean with custom configurations
 */
@Configuration
@Slf4j
public class RestTemplateConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder restTemplateBuilder) {
        return restTemplateBuilder
                .setConnectTimeout(Duration.ofSeconds(5))
                .setReadTimeout(Duration.ofSeconds(5))
                .defaultHeader(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .errorHandler(new CustomRestTemplateErrorHandler())
                .build();
    }

    /**
     * Custom error handler for RestTemplate
     */
    private static class CustomRestTemplateErrorHandler implements ResponseErrorHandler {

        @Override
        public boolean hasError(@SuppressWarnings("null") ClientHttpResponse response) throws IOException {
            return response.getStatusCode().is4xxClientError() || 
                   response.getStatusCode().is5xxServerError();
        }

        @Override
        public void handleError(@SuppressWarnings("null") ClientHttpResponse response) throws IOException {
            @SuppressWarnings("resource")
            String responseBody = new BufferedReader(new InputStreamReader(response.getBody()))
                    .lines().collect(Collectors.joining("\n"));

            String errorMessage = String.format("Error during API call: Status Code: %s, Headers: %s, Body: %s",
                    response.getStatusCode(), response.getHeaders(), responseBody);

            log.error(errorMessage);

            if (response.getStatusCode().is5xxServerError()) {
                throw new RuntimeException("Server error calling Python API: " + errorMessage);
            } else if (response.getStatusCode().is4xxClientError()) {
                throw new RuntimeException("Client error calling Python API: " + errorMessage);
            }
        }
    }
}
