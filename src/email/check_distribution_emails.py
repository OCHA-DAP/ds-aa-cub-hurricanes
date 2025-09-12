# Test script for validating distribution list emails
from src.email.utils import is_valid_email, get_distribution_list


def test_distribution_list_emails():
    """Test all emails in the distribution list for validity."""
    print("📧 Testing distribution list email validity...")
    
    # Load the distribution list
    distribution_list = get_distribution_list()
    
    # Test each email
    valid_emails = []
    invalid_emails = []
    
    for idx, row in distribution_list.iterrows():
        email = row['email']
        name = row.get('name', 'Unknown')
        
        if is_valid_email(email):
            valid_emails.append((name, email))
        else:
            invalid_emails.append((name, email))
    
    # Print results
    print(f"\n✅ Valid emails ({len(valid_emails)}):")
    for name, email in valid_emails:
        print(f"   • {name}: {email}")
    
    if invalid_emails:
        print(f"\n❌ Invalid emails ({len(invalid_emails)}):")
        for name, email in invalid_emails:
            print(f"   • {name}: {email}")
    else:
        print(f"\n🎉 All {len(valid_emails)} emails are valid!")
    
    # Summary
    total = len(distribution_list)
    print(f"\n📊 Summary: {len(valid_emails)}/{total} valid emails")
    
    return {
        'valid': valid_emails,
        'invalid': invalid_emails,
        'total': total
    }


if __name__ == "__main__":
    results = test_distribution_list_emails()