package com.devicepricing;

import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.servlet.mvc.method.annotation.ResponseEntityExceptionHandler;

import jakarta.validation.ConstraintViolationException;

import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.http.converter.HttpMessageNotReadableException;

import java.util.HashMap;
import java.util.Map;

/**
 * Global exception handler for handling all exceptions across the application.
 */
@ControllerAdvice
public class GlobalExceptionHandler extends ResponseEntityExceptionHandler {

    /**
     * Handle validation errors and return a detailed response.
     *
     * @param ex      The exception thrown when validation fails.
     * @param headers The HTTP headers to be returned.
     * @param status  The HTTP status to be returned.
     * @param request The current request.
     * @return ResponseEntity with a detailed error response.
     */
    @SuppressWarnings("null")
    @Override
    protected ResponseEntity<Object> handleMethodArgumentNotValid(MethodArgumentNotValidException ex,
                                                                  HttpHeaders headers,
                                                                  HttpStatusCode status,
                                                                  WebRequest request) {
        Map<String, String> errors = new HashMap<>();
        ex.getBindingResult().getFieldErrors().forEach(error ->
                errors.put(error.getField(), error.getDefaultMessage()));

        ErrorResponse errorResponse = new ErrorResponse(400, "Validation Failed");
        errorResponse.setValidationErrors(errors);
        return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
    }
    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<ErrorResponse> handleDataIntegrityViolationException(DataIntegrityViolationException ex) {
        // Check if it's a unique constraint violation
        if (ex.getCause() instanceof ConstraintViolationException) {
            ErrorResponse errorResponse = new ErrorResponse(
                409,
                "A device with this ID already exists"
            );
            return new ResponseEntity<>(errorResponse, HttpStatus.CONFLICT);
        }
        
    // Handle other data integrity violations
    ErrorResponse errorResponse = new ErrorResponse(
        400,
        "Data integrity violation: " + ex.getMessage()
    );
    return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
}
    /**
     * Handle resource not found exceptions.
     *
     * @param ex The ResourceNotFoundException thrown when a resource is not found.
     * @return ResponseEntity with a detailed error response.
     */
    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleResourceNotFoundException(ResourceNotFoundException ex) {
        ErrorResponse errorResponse = new ErrorResponse(404, ex.getMessage());
        return new ResponseEntity<>(errorResponse, HttpStatus.NOT_FOUND);
    }

    /**
     * Handle all other exceptions.
     *
     * @param ex The exception thrown.
     * @return ResponseEntity with a detailed error response.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleAllExceptions(Exception ex) {
        ErrorResponse errorResponse = new ErrorResponse(500, "An unexpected error occurred: " + ex.getMessage());
        return new ResponseEntity<>(errorResponse, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    /**
     * Handle HttpRequestMethodNotSupportedException.
     *
     * @param ex      The exception thrown.
     * @param headers The HTTP headers.
     * @param status  The HTTP status.
     * @param request The current request.
     * @return ResponseEntity with a detailed error response.
     */
    @SuppressWarnings("null")
    @Override
    protected ResponseEntity<Object> handleHttpRequestMethodNotSupported(HttpRequestMethodNotSupportedException ex,
                                                                         HttpHeaders headers,
                                                                         HttpStatusCode status,
                                                                         WebRequest request) {
        ErrorResponse errorResponse = new ErrorResponse(405, "Method not allowed: " + ex.getMethod());
        return new ResponseEntity<>(errorResponse, HttpStatus.METHOD_NOT_ALLOWED);
    }

    /**
     * Handle HttpMessageNotReadableException for invalid request body.
     *
     * @param ex The exception thrown when the request body is unreadable.
     * @param headers The HTTP headers.
     * @param status The HTTP status.
     * @param request The current request.
     * @return ResponseEntity with a detailed error response.
     */
    @SuppressWarnings("null")
    @Override
    protected ResponseEntity<Object> handleHttpMessageNotReadable(HttpMessageNotReadableException ex,
                                                                  HttpHeaders headers,
                                                                  HttpStatusCode status,
                                                                  WebRequest request) {
        ErrorResponse errorResponse = new ErrorResponse(400, "Invalid request body: " + ex.getMessage());
        return new ResponseEntity<>(errorResponse, HttpStatus.BAD_REQUEST);
    }
}
