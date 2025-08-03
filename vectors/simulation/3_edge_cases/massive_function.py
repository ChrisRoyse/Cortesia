"""
File with extremely long functions to test chunking strategies.
This simulates legacy code with poor structure and very long functions.
"""

import re
import json
import datetime
import hashlib
from typing import Dict, List, Any, Optional, Union

class LegacyDataProcessor:
    """A legacy data processor with massive functions that need refactoring."""
    
    def __init__(self):
        self.data_cache = {}
        self.error_log = []
        self.processing_stats = {}
        
    def massive_data_processing_function(self, raw_data: List[Dict[str, Any]], 
                                       config: Dict[str, Any], 
                                       output_format: str = "json") -> Union[str, Dict, List]:
        """
        MASSIVE LEGACY FUNCTION - NEEDS REFACTORING
        
        This function processes various types of data through multiple complex steps.
        It violates single responsibility principle and is extremely difficult to maintain.
        Used for testing how indexing systems handle very long functions.
        
        WARNING: This is intentionally bad code for testing purposes!
        """
        # Initialize variables and counters
        processed_records = []
        error_count = 0
        warning_count = 0
        total_processed = 0
        start_time = datetime.datetime.now()
        validation_errors = []
        transformation_log = []
        data_quality_issues = []
        performance_metrics = {}
        temporary_storage = {}
        lookup_tables = {}
        regex_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$',
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
            'zip_code': r'^\d{5}(-\d{4})?$',
            'ip_address': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            'url': r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            'time_24h': r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$',
            'currency': r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$'
        }
        
        # Step 1: Initial data validation and preprocessing
        print("Starting massive data processing function...")
        print(f"Processing {len(raw_data)} records...")
        
        for record_index, record in enumerate(raw_data):
            try:
                # Validate record structure
                if not isinstance(record, dict):
                    error_count += 1
                    validation_errors.append(f"Record {record_index}: Invalid record type, expected dict, got {type(record)}")
                    continue
                
                # Check for required fields
                required_fields = config.get('required_fields', [])
                missing_fields = []
                for field in required_fields:
                    if field not in record or record[field] is None or record[field] == '':
                        missing_fields.append(field)
                
                if missing_fields:
                    warning_count += 1
                    validation_errors.append(f"Record {record_index}: Missing required fields: {missing_fields}")
                    if config.get('strict_validation', False):
                        continue
                
                # Data type validation
                field_types = config.get('field_types', {})
                for field_name, expected_type in field_types.items():
                    if field_name in record:
                        field_value = record[field_name]
                        if expected_type == 'string' and not isinstance(field_value, str):
                            try:
                                record[field_name] = str(field_value)
                                transformation_log.append(f"Record {record_index}: Converted {field_name} to string")
                            except Exception as e:
                                validation_errors.append(f"Record {record_index}: Cannot convert {field_name} to string: {e}")
                                continue
                        elif expected_type == 'integer' and not isinstance(field_value, int):
                            try:
                                record[field_name] = int(float(field_value))
                                transformation_log.append(f"Record {record_index}: Converted {field_name} to integer")
                            except (ValueError, TypeError) as e:
                                validation_errors.append(f"Record {record_index}: Cannot convert {field_name} to integer: {e}")
                                continue
                        elif expected_type == 'float' and not isinstance(field_value, (int, float)):
                            try:
                                record[field_name] = float(field_value)
                                transformation_log.append(f"Record {record_index}: Converted {field_name} to float")
                            except (ValueError, TypeError) as e:
                                validation_errors.append(f"Record {record_index}: Cannot convert {field_name} to float: {e}")
                                continue
                        elif expected_type == 'boolean' and not isinstance(field_value, bool):
                            try:
                                if isinstance(field_value, str):
                                    record[field_name] = field_value.lower() in ['true', '1', 'yes', 'on', 'enabled']
                                else:
                                    record[field_name] = bool(field_value)
                                transformation_log.append(f"Record {record_index}: Converted {field_name} to boolean")
                            except Exception as e:
                                validation_errors.append(f"Record {record_index}: Cannot convert {field_name} to boolean: {e}")
                                continue
                
                # Step 2: Field-specific validation using regex patterns
                validation_fields = config.get('validation_fields', {})
                for field_name, validation_type in validation_fields.items():
                    if field_name in record and validation_type in regex_patterns:
                        field_value = str(record[field_name])
                        pattern = regex_patterns[validation_type]
                        if not re.match(pattern, field_value):
                            validation_errors.append(f"Record {record_index}: Invalid {validation_type} format for {field_name}: {field_value}")
                            if config.get('strict_validation', False):
                                continue
                            else:
                                # Try to clean/fix common issues
                                if validation_type == 'phone':
                                    # Remove common formatting characters
                                    cleaned_phone = re.sub(r'[^\d+]', '', field_value)
                                    if len(cleaned_phone) == 10:
                                        cleaned_phone = f"({cleaned_phone[:3]}) {cleaned_phone[3:6]}-{cleaned_phone[6:]}"
                                        record[field_name] = cleaned_phone
                                        transformation_log.append(f"Record {record_index}: Cleaned phone number format")
                                elif validation_type == 'email':
                                    # Convert to lowercase and remove spaces
                                    cleaned_email = field_value.lower().strip()
                                    if re.match(regex_patterns['email'], cleaned_email):
                                        record[field_name] = cleaned_email
                                        transformation_log.append(f"Record {record_index}: Cleaned email format")
                                elif validation_type == 'ssn':
                                    # Add dashes if missing
                                    cleaned_ssn = re.sub(r'[^\d]', '', field_value)
                                    if len(cleaned_ssn) == 9:
                                        cleaned_ssn = f"{cleaned_ssn[:3]}-{cleaned_ssn[3:5]}-{cleaned_ssn[5:]}"
                                        record[field_name] = cleaned_ssn
                                        transformation_log.append(f"Record {record_index}: Cleaned SSN format")
                
                # Step 3: Data enrichment and lookup operations
                enrichment_config = config.get('enrichment', {})
                if enrichment_config.get('enable_lookups', False):
                    # Geographic enrichment
                    if 'zip_code' in record:
                        zip_code = record['zip_code']
                        if zip_code not in lookup_tables:
                            # Simulate expensive lookup operation
                            geo_data = self._lookup_geographic_data(zip_code)
                            lookup_tables[zip_code] = geo_data
                        
                        geo_info = lookup_tables[zip_code]
                        if geo_info:
                            record['city'] = geo_info.get('city', '')
                            record['state'] = geo_info.get('state', '')
                            record['county'] = geo_info.get('county', '')
                            record['timezone'] = geo_info.get('timezone', '')
                            record['area_code'] = geo_info.get('area_code', '')
                            transformation_log.append(f"Record {record_index}: Added geographic enrichment")
                    
                    # Industry classification
                    if 'company_name' in record:
                        company_name = record['company_name']
                        industry_data = self._classify_industry(company_name)
                        if industry_data:
                            record['industry_category'] = industry_data.get('category', 'Unknown')
                            record['industry_subcategory'] = industry_data.get('subcategory', 'Unknown')
                            record['company_size_estimate'] = industry_data.get('size_estimate', 'Unknown')
                            transformation_log.append(f"Record {record_index}: Added industry classification")
                    
                    # Credit score enrichment (simulated)
                    if 'ssn' in record and enrichment_config.get('enable_credit_lookup', False):
                        ssn = record['ssn']
                        credit_data = self._lookup_credit_data(ssn)
                        if credit_data:
                            record['credit_score_range'] = credit_data.get('score_range', 'Unknown')
                            record['credit_history_length'] = credit_data.get('history_length', 'Unknown')
                            transformation_log.append(f"Record {record_index}: Added credit information")
                
                # Step 4: Data quality scoring
                quality_score = 100
                quality_issues = []
                
                # Completeness check
                total_fields = len(record)
                empty_fields = sum(1 for v in record.values() if v is None or v == '')
                completeness_ratio = (total_fields - empty_fields) / total_fields if total_fields > 0 else 0
                if completeness_ratio < 0.8:
                    quality_score -= 20
                    quality_issues.append(f"Low completeness: {completeness_ratio:.2%}")
                
                # Consistency checks
                if 'birth_date' in record and 'age' in record:
                    try:
                        birth_date = datetime.datetime.strptime(record['birth_date'], '%Y-%m-%d')
                        calculated_age = (datetime.datetime.now() - birth_date).days // 365
                        reported_age = int(record['age'])
                        if abs(calculated_age - reported_age) > 2:
                            quality_score -= 10
                            quality_issues.append(f"Age inconsistency: calculated {calculated_age}, reported {reported_age}")
                    except (ValueError, TypeError):
                        pass
                
                # Format consistency
                inconsistent_formats = 0
                if 'phone' in record and 'phone_2' in record:
                    phone1_digits = re.sub(r'[^\d]', '', str(record['phone']))
                    phone2_digits = re.sub(r'[^\d]', '', str(record['phone_2']))
                    if len(phone1_digits) == len(phone2_digits) == 10:
                        phone1_format = re.sub(r'\d', 'X', str(record['phone']))
                        phone2_format = re.sub(r'\d', 'X', str(record['phone_2']))
                        if phone1_format != phone2_format:
                            inconsistent_formats += 1
                
                if inconsistent_formats > 0:
                    quality_score -= inconsistent_formats * 5
                    quality_issues.append(f"Inconsistent formatting in {inconsistent_formats} field pairs")
                
                # Duplicate detection within record
                value_frequency = {}
                for field_name, field_value in record.items():
                    if isinstance(field_value, (str, int, float)) and field_value != '':
                        value_str = str(field_value).lower().strip()
                        if len(value_str) > 3:  # Only check meaningful values
                            value_frequency[value_str] = value_frequency.get(value_str, 0) + 1
                
                duplicate_values = [v for v, count in value_frequency.items() if count > 1]
                if duplicate_values:
                    quality_score -= len(duplicate_values) * 3
                    quality_issues.append(f"Duplicate values: {duplicate_values[:3]}...")
                
                record['data_quality_score'] = max(0, quality_score)
                if quality_issues:
                    record['data_quality_issues'] = quality_issues
                    data_quality_issues.extend(quality_issues)
                
                # Step 5: Advanced transformations
                transformation_config = config.get('transformations', {})
                
                # Name standardization
                if transformation_config.get('standardize_names', False):
                    name_fields = ['first_name', 'last_name', 'middle_name', 'company_name']
                    for field in name_fields:
                        if field in record and isinstance(record[field], str):
                            original_value = record[field]
                            # Title case conversion
                            standardized = original_value.title()
                            # Handle special cases
                            standardized = re.sub(r"\bMc([a-z])", r"Mc\1", standardized)
                            standardized = re.sub(r"\bO'([a-z])", r"O'\1", standardized)
                            standardized = re.sub(r"\bVan ([a-z])", r"van \1", standardized)
                            standardized = re.sub(r"\bDe ([a-z])", r"de \1", standardized)
                            if standardized != original_value:
                                record[field] = standardized
                                transformation_log.append(f"Record {record_index}: Standardized {field}")
                
                # Address standardization
                if transformation_config.get('standardize_addresses', False):
                    address_fields = ['street_address', 'address_line_1', 'address_line_2']
                    for field in address_fields:
                        if field in record and isinstance(record[field], str):
                            original_address = record[field]
                            standardized_address = original_address.upper()
                            # Common abbreviations
                            address_abbreviations = {
                                r'\bSTREET\b': 'ST',
                                r'\bAVENUE\b': 'AVE',
                                r'\bBOULEVARD\b': 'BLVD',
                                r'\bROAD\b': 'RD',
                                r'\bLANE\b': 'LN',
                                r'\bDRIVE\b': 'DR',
                                r'\bCOURT\b': 'CT',
                                r'\bPLACE\b': 'PL',
                                r'\bCIRCLE\b': 'CIR',
                                r'\bPARKWAY\b': 'PKWY',
                                r'\bAPARTMENT\b': 'APT',
                                r'\bSUITE\b': 'STE',
                                r'\bNORTH\b': 'N',
                                r'\bSOUTH\b': 'S',
                                r'\bEAST\b': 'E',
                                r'\bWEST\b': 'W'
                            }
                            for pattern, replacement in address_abbreviations.items():
                                standardized_address = re.sub(pattern, replacement, standardized_address)
                            
                            if standardized_address != original_address.upper():
                                record[field] = standardized_address
                                transformation_log.append(f"Record {record_index}: Standardized address {field}")
                
                # Currency formatting
                if transformation_config.get('format_currency', False):
                    currency_fields = ['salary', 'income', 'revenue', 'budget', 'amount']
                    for field in currency_fields:
                        if field in record:
                            try:
                                value = record[field]
                                if isinstance(value, str):
                                    # Remove currency symbols and commas
                                    numeric_value = re.sub(r'[^\d.-]', '', value)
                                    if numeric_value:
                                        record[field] = float(numeric_value)
                                elif isinstance(value, (int, float)):
                                    record[field] = float(value)
                                transformation_log.append(f"Record {record_index}: Formatted currency {field}")
                            except (ValueError, TypeError):
                                validation_errors.append(f"Record {record_index}: Cannot format currency {field}")
                
                # Step 6: Custom business rules
                business_rules = config.get('business_rules', {})
                
                # Age validation
                if business_rules.get('validate_age_ranges', False):
                    if 'age' in record:
                        age = record.get('age')
                        try:
                            age_int = int(age)
                            if age_int < 0 or age_int > 150:
                                validation_errors.append(f"Record {record_index}: Invalid age: {age_int}")
                                if business_rules.get('fix_invalid_ages', False):
                                    record['age'] = None
                                    transformation_log.append(f"Record {record_index}: Cleared invalid age")
                        except (ValueError, TypeError):
                            validation_errors.append(f"Record {record_index}: Non-numeric age: {age}")
                
                # Income validation
                if business_rules.get('validate_income', False):
                    income_fields = ['salary', 'income', 'annual_income']
                    for field in income_fields:
                        if field in record:
                            try:
                                income = float(record[field])
                                if income < 0:
                                    validation_errors.append(f"Record {record_index}: Negative income in {field}")
                                elif income > 10000000:  # 10 million
                                    validation_errors.append(f"Record {record_index}: Unusually high income in {field}: {income}")
                            except (ValueError, TypeError):
                                pass
                
                # Relationship consistency
                if business_rules.get('validate_relationships', False):
                    if 'marital_status' in record and 'spouse_name' in record:
                        marital_status = str(record['marital_status']).lower()
                        spouse_name = str(record['spouse_name']).strip()
                        if marital_status in ['single', 'divorced', 'widowed'] and spouse_name:
                            validation_errors.append(f"Record {record_index}: Inconsistent marital status and spouse name")
                        elif marital_status == 'married' and not spouse_name:
                            validation_errors.append(f"Record {record_index}: Married status but no spouse name")
                
                # Step 7: Generate derived fields
                derivation_config = config.get('derived_fields', {})
                
                # Full name generation
                if derivation_config.get('generate_full_name', False):
                    name_parts = []
                    for part in ['first_name', 'middle_name', 'last_name']:
                        if part in record and record[part]:
                            name_parts.append(str(record[part]).strip())
                    if name_parts:
                        record['full_name'] = ' '.join(name_parts)
                        transformation_log.append(f"Record {record_index}: Generated full name")
                
                # Age calculation from birth date
                if derivation_config.get('calculate_age_from_birth_date', False):
                    if 'birth_date' in record and record['birth_date']:
                        try:
                            birth_date = datetime.datetime.strptime(str(record['birth_date']), '%Y-%m-%d')
                            age = (datetime.datetime.now() - birth_date).days // 365
                            record['calculated_age'] = age
                            transformation_log.append(f"Record {record_index}: Calculated age from birth date")
                        except (ValueError, TypeError):
                            pass
                
                # Email domain extraction
                if derivation_config.get('extract_email_domain', False):
                    email_fields = ['email', 'work_email', 'personal_email']
                    for field in email_fields:
                        if field in record and record[field]:
                            email = str(record[field])
                            if '@' in email:
                                domain = email.split('@')[-1].lower()
                                record[f'{field}_domain'] = domain
                                transformation_log.append(f"Record {record_index}: Extracted domain from {field}")
                
                # Geographic coordinates (simulated)
                if derivation_config.get('geocode_addresses', False):
                    if 'street_address' in record and 'city' in record and 'state' in record:
                        # Simulate geocoding
                        address_hash = hashlib.md5(
                            f"{record['street_address']}{record['city']}{record['state']}".encode()
                        ).hexdigest()
                        # Generate pseudo-random coordinates based on hash
                        lat_offset = int(address_hash[:8], 16) % 180 - 90
                        lon_offset = int(address_hash[8:16], 16) % 360 - 180
                        record['latitude'] = lat_offset + (int(address_hash[16:18], 16) % 100) / 100
                        record['longitude'] = lon_offset + (int(address_hash[18:20], 16) % 100) / 100
                        transformation_log.append(f"Record {record_index}: Added geocoded coordinates")
                
                # Step 8: Data masking and privacy protection
                privacy_config = config.get('privacy', {})
                
                if privacy_config.get('mask_sensitive_data', False):
                    sensitive_fields = privacy_config.get('sensitive_fields', ['ssn', 'credit_card', 'bank_account'])
                    for field in sensitive_fields:
                        if field in record and record[field]:
                            original_value = str(record[field])
                            if field == 'ssn':
                                # Mask all but last 4 digits
                                masked_value = 'XXX-XX-' + original_value[-4:]
                                record[field + '_masked'] = masked_value
                            elif field == 'credit_card':
                                # Mask all but last 4 digits
                                digits_only = re.sub(r'[^\d]', '', original_value)
                                if len(digits_only) >= 4:
                                    masked_value = '*' * (len(digits_only) - 4) + digits_only[-4:]
                                    record[field + '_masked'] = masked_value
                            elif field == 'bank_account':
                                # Mask all but last 2 digits
                                digits_only = re.sub(r'[^\d]', '', original_value)
                                if len(digits_only) >= 2:
                                    masked_value = '*' * (len(digits_only) - 2) + digits_only[-2:]
                                    record[field + '_masked'] = masked_value
                            
                            if privacy_config.get('remove_original', False):
                                del record[field]
                            
                            transformation_log.append(f"Record {record_index}: Masked sensitive field {field}")
                
                # Step 9: Data aggregation and statistics
                aggregation_config = config.get('aggregation', {})
                
                if aggregation_config.get('calculate_statistics', False):
                    numeric_fields = []
                    for field_name, field_value in record.items():
                        try:
                            float(field_value)
                            numeric_fields.append(field_name)
                        except (ValueError, TypeError):
                            pass
                    
                    if numeric_fields:
                        record['numeric_field_count'] = len(numeric_fields)
                        record['numeric_fields'] = numeric_fields
                        transformation_log.append(f"Record {record_index}: Calculated numeric field statistics")
                
                # Step 10: Final validation and cleanup
                final_validation_config = config.get('final_validation', {})
                
                # Remove empty fields if configured
                if final_validation_config.get('remove_empty_fields', False):
                    fields_to_remove = []
                    for field_name, field_value in record.items():
                        if field_value is None or field_value == '' or field_value == []:
                            fields_to_remove.append(field_name)
                    
                    for field_name in fields_to_remove:
                        del record[field_name]
                    
                    if fields_to_remove:
                        transformation_log.append(f"Record {record_index}: Removed {len(fields_to_remove)} empty fields")
                
                # Add processing metadata
                if config.get('add_processing_metadata', False):
                    record['_processing_timestamp'] = datetime.datetime.now().isoformat()
                    record['_record_index'] = record_index
                    record['_transformation_count'] = len([log for log in transformation_log if f"Record {record_index}" in log])
                    record['_validation_error_count'] = len([err for err in validation_errors if f"Record {record_index}" in err])
                
                # Add record to processed list
                processed_records.append(record)
                total_processed += 1
                
                # Progress reporting
                if total_processed % 1000 == 0:
                    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                    records_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
                    print(f"Processed {total_processed} records ({records_per_second:.1f} records/sec)")
                
            except Exception as e:
                error_count += 1
                self.error_log.append(f"Error processing record {record_index}: {str(e)}")
                print(f"Error processing record {record_index}: {str(e)}")
        
        # Final processing and output generation
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Generate processing statistics
        self.processing_stats = {
            'total_input_records': len(raw_data),
            'successfully_processed': total_processed,
            'error_count': error_count,
            'warning_count': warning_count,
            'processing_time_seconds': processing_time,
            'records_per_second': total_processed / processing_time if processing_time > 0 else 0,
            'validation_errors': len(validation_errors),
            'transformations_applied': len(transformation_log),
            'data_quality_issues': len(data_quality_issues),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
        
        print(f"Processing complete: {total_processed} records processed in {processing_time:.2f} seconds")
        print(f"Errors: {error_count}, Warnings: {warning_count}")
        
        # Generate output based on requested format
        if output_format.lower() == 'json':
            output_data = {
                'metadata': self.processing_stats,
                'records': processed_records,
                'validation_errors': validation_errors[:100],  # Limit to first 100 errors
                'transformation_log': transformation_log[:100],  # Limit to first 100 transformations
                'data_quality_summary': {
                    'total_issues': len(data_quality_issues),
                    'unique_issues': len(set(data_quality_issues)),
                    'most_common_issues': self._get_most_common_issues(data_quality_issues)
                }
            }
            return json.dumps(output_data, indent=2, default=str)
        
        elif output_format.lower() == 'csv':
            # Convert to CSV format (simplified)
            if processed_records:
                all_fields = set()
                for record in processed_records:
                    all_fields.update(record.keys())
                
                csv_lines = [','.join(sorted(all_fields))]
                for record in processed_records:
                    row_values = []
                    for field in sorted(all_fields):
                        value = record.get(field, '')
                        # Escape commas and quotes
                        if isinstance(value, str) and (',' in value or '"' in value):
                            value = '"' + value.replace('"', '""') + '"'
                        row_values.append(str(value))
                    csv_lines.append(','.join(row_values))
                
                return '\n'.join(csv_lines)
            else:
                return "No records to export"
        
        elif output_format.lower() == 'summary':
            summary_data = {
                'processing_summary': self.processing_stats,
                'sample_records': processed_records[:5],  # First 5 records
                'error_summary': validation_errors[:10],  # First 10 errors
                'transformation_summary': transformation_log[:10]  # First 10 transformations
            }
            return summary_data
        
        else:
            # Return raw processed records
            return processed_records
    
    def _lookup_geographic_data(self, zip_code: str) -> Dict[str, str]:
        """Simulate geographic data lookup."""
        # This would normally be a database or API call
        fake_data = {
            '12345': {'city': 'New York', 'state': 'NY', 'county': 'Manhattan', 'timezone': 'EST', 'area_code': '212'},
            '90210': {'city': 'Beverly Hills', 'state': 'CA', 'county': 'Los Angeles', 'timezone': 'PST', 'area_code': '310'},
            '60601': {'city': 'Chicago', 'state': 'IL', 'county': 'Cook', 'timezone': 'CST', 'area_code': '312'}
        }
        return fake_data.get(zip_code, {'city': 'Unknown', 'state': 'Unknown', 'county': 'Unknown', 'timezone': 'Unknown', 'area_code': 'Unknown'})
    
    def _classify_industry(self, company_name: str) -> Dict[str, str]:
        """Simulate industry classification."""
        # This would normally use ML models or industry databases
        if 'bank' in company_name.lower() or 'financial' in company_name.lower():
            return {'category': 'Financial Services', 'subcategory': 'Banking', 'size_estimate': 'Large'}
        elif 'tech' in company_name.lower() or 'software' in company_name.lower():
            return {'category': 'Technology', 'subcategory': 'Software', 'size_estimate': 'Medium'}
        elif 'health' in company_name.lower() or 'medical' in company_name.lower():
            return {'category': 'Healthcare', 'subcategory': 'Medical Services', 'size_estimate': 'Medium'}
        else:
            return {'category': 'Other', 'subcategory': 'Unknown', 'size_estimate': 'Unknown'}
    
    def _lookup_credit_data(self, ssn: str) -> Dict[str, str]:
        """Simulate credit data lookup."""
        # This would normally be a secure credit bureau API call
        return {
            'score_range': '700-750',
            'history_length': '10+ years'
        }
    
    def _get_most_common_issues(self, issues: List[str]) -> List[Dict[str, Any]]:
        """Get most common data quality issues."""
        issue_counts = {}
        for issue in issues:
            # Extract issue type from the issue string
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Sort by frequency and return top 5
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'issue_type': issue, 'count': count} for issue, count in sorted_issues[:5]]